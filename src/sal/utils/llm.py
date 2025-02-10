"""HuggingFace Transformers-based LLM implementation."""

import fla
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import torch
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any, Literal
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.append('/n/home01/mkulkarni/projects/inference-scaling/3rdparty/MambaInLlama')
from mamba_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper
from mamba2_inference.hybrid_wrapper import MambaTransformerHybridModelWrapper as Mamba2TransformerHybridModelWrapper

@dataclass
class SamplingParams:
    """Parameters for text generation, similar to vLLM's SamplingParams."""
    
    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 0
    min_p: float = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: bool = False
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    eos_token_id: Optional[int] = None
    max_tokens: int = 16
    beam_width: Optional[int] = None
    num_iterations: Optional[int] = None
    lookahead: Optional[int] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    include_stop_str_in_output: bool = False
    
    @classmethod
    def from_config(cls, config) -> "SamplingParams":
        """Create SamplingParams from a Config instance."""
        params = cls(
            n=config.n,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
            use_beam_search=config.approach in ["beam_search", "dvts"],
        )
        
        if config.approach in ["beam_search", "dvts"]:
            params.beam_width = config.beam_width
            params.num_iterations = config.num_iterations
            params.lookahead = config.lookahead
            params.best_of = config.beam_width
            
        return params

class RequestOutput:
    """Output of a generation request, similar to vLLM's RequestOutput."""
    
    def __init__(
        self,
        text: str,
        tokens: List[int],
        logprobs: Optional[List[float]] = None,
        finish_reason: Optional[str] = None
    ):
       self.text = text
       self.tokens = tokens
       self.logprobs = logprobs
       self.finish_reason = finish_reason
       self.prompt_token_ids = []
       self.outputs = [self]
       self.stop_reason = finish_reason

    @property
    def token_ids(self):
        return self.tokens

class LLM:
    """LLM implementation using Hugging Face Transformers."""
    
    def __init__(
        self,
        model: str,
        trust_remote_code: bool = True,
        device_map: str = "auto",
        seed: Optional[int] = None,
        hybrid: bool = False
    ):
        if seed is not None:
            torch.manual_seed(seed)
            
        # Initialize distributed training only if environment is properly set up
        self.distributed = False
        if torch.cuda.device_count() > 1 and "LOCAL_RANK" in os.environ:
            try:
                local_rank = int(os.environ["LOCAL_RANK"])
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")
                torch.cuda.set_device(local_rank)
                self.distributed = True
                print(f"Initialized distributed training with local rank {local_rank}")
            except Exception as e:
                print(f"Failed to initialize distributed training: {e}")
                print("Falling back to single-GPU or CPU mode")
            
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
        
        # Load model with device map for multiple GPUs
        if hybrid:
            if "mamba2" in model.lower():
                self.model = Mamba2TransformerHybridModelWrapper.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
            else:
                self.model = MambaTransformerHybridModelWrapper.from_pretrained(
                    model,
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                device_map="balanced" if torch.cuda.device_count() > 1 else "auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=trust_remote_code,
                offload_folder="offload",
            )
        self.model = torch.compile(self.model)
        
        # Wrap model in DDP if using multiple GPUs
        if torch.cuda.device_count() > 1:
            self.model = DistributedDataParallel(
                self.model, 
                device_ids=[local_rank],
                output_device=local_rank
            )

        print(f"Model loaded across {torch.cuda.device_count()} GPUs")

        self.model.eval() 

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Enable token logging
        self.model.config.output_scores = True
        self.model.config.return_dict_in_generate = True
        
    def get_tokenizer(self):
        """Return the tokenizer."""
        return self.tokenizer
    
    def set_chat_template(self, template: str):
        """Set a custom chat template."""
        self.tokenizer.chat_template = template
    
    def generate(
        self,
        prompts: Union[str, List[str]],
        sampling_params: Optional[SamplingParams] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[RequestOutput]:
        """Generate completions for the prompt(s)."""
        
        if sampling_params is None:
            sampling_params = SamplingParams()
            
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # # Apply chat template if system prompt is provided
        # if system_prompt:
        #     formatted_prompts = []
        #     for prompt in prompts:
        #         messages = [
        #             {"role": "system", "content": system_prompt},
        #             {"role": "user", "content": prompt}
        #         ]
        #         formatted_prompts.append(self.tokenizer.apply_chat_template(
        #             messages,
        #             tokenize=False,
        #             add_generation_prompt=True
        #         ))
        #     prompts = formatted_prompts
            
        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Prepare generation kwargs
        # In class LLM, in the generate method, replace the generation_kwargs block with:
        generation_kwargs = {
            "max_new_tokens": sampling_params.max_tokens,
            "num_return_sequences": sampling_params.n,
            "temperature": sampling_params.temperature,
            "top_p": sampling_params.top_p,
            "top_k": sampling_params.top_k,
            "min_p": sampling_params.min_p,
            "do_sample": True,  # Force sampling to be enabled
            "output_scores": True,
            "return_dict_in_generate": True,
            # "eos_token_id": self.tokenizer.eos_token_id
        }
        
        if sampling_params.use_beam_search:
            generation_kwargs.update({
                "num_beams": sampling_params.beam_width or sampling_params.best_of or 4,
                "early_stopping": sampling_params.early_stopping,
                "length_penalty": sampling_params.length_penalty,
                "stop": sampling_params.stop,
                # "include_stop_str_in_output": sampling_params.include_stop_str_in_output,
            })
            
        if sampling_params.stop_token_ids:
            generation_kwargs["eos_token_id"] = sampling_params.stop_token_ids

        generation_kwargs['eos_token_id'] = self.tokenizer.eos_token_id
        # Generate
        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)
        
        # Process outputs
        results = []
        for sequence, scores in zip(outputs.sequences, outputs.scores):
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = sequence[input_length:]
            prompt_tokens = sequence[:input_length]
            
            # Get logprobs if requested
            logprobs = None
            if sampling_params.logprobs is not None:
                logprobs = [score.max().item() for score in scores]
            
            # Decode text
            text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
                spaces_between_special_tokens=True
            )
            
            # Determine finish reason
            finish_reason = "length"
            if generated_tokens[-1] == self.tokenizer.eos_token_id:
                finish_reason = "stop"
            
            result = RequestOutput(
                text=text,
                tokens=generated_tokens.tolist(),
                logprobs=logprobs,
                finish_reason=finish_reason
            )
            result.prompt_token_ids = prompt_tokens.tolist()
            results.append(result)
        
        return results
