#!/usr/bin/env python3
#  2022, University of Stuttgart;  Pavel Denisov
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Hugging Face Transformers Decoder."""

import copy
import logging
from typing import Tuple

import torch
import torch.nn.functional as F
from typeguard import check_argument_types

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

try:
    from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
    from peft import prepare_model_for_int8_training

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

try:
    from openprompt import PromptForGeneration
    from openprompt.plms import load_plm
    from openprompt.prompts.prefix_tuning_template import PrefixTuningTemplate

    is_openprompt_available = True
except ImportError:
    is_openprompt_available = False


class HuggingFaceTransformersDecoder(AbsDecoder):
    """Hugging Face Transformers Decoder.

    Args:
        encoder_output_size: dimension of encoder attention
        model_name_or_path: Hugging Face Transformers model name
    """

    def __init__(
        self,
        vocab_size: int,
        encoder_output_size: int,
        model_name_or_path: str,
        causal_lm: bool = False,
        prefix: str = "",
        postfix: str = "",
        prefix_tuning_template_path: str = "",
        prefix_tuning_template_plm: str = "",
        prefix_tuning_template_text: str = "",
        load_in_8bit: bool = False,
        torch_compile: bool = False,
    ):
        assert check_argument_types()
        super().__init__()

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        if prefix_tuning_template_path != "":
            if not is_openprompt_available:
                raise ImportError(
                    "`openprompt` is not available. Please install it via `pip install"
                    " openprompt`."
                )

            self.causal_lm = True

            plm, tokenizer, model_config, WrapperClass = load_plm(
                prefix_tuning_template_plm, model_name_or_path
            )
            mytemplate = PrefixTuningTemplate(
                model=plm,
                tokenizer=tokenizer,
                text=prefix_tuning_template_text,
                mid_dim=2048,
                num_token=10,
            )
            self.decoder = PromptForGeneration(
                plm=plm,
                template=mytemplate,
                freeze_plm=True,
                tokenizer=tokenizer,
                plm_eval_mode=False,
            )

            self.decoder.load_state_dict(torch.load(prefix_tuning_template_path))

            self.decoder.plm.resize_token_embeddings(vocab_size)

            self.decoder_pretrained_params = copy.deepcopy(self.decoder.state_dict())
            self.lm_head_pretrained_params = None

            if encoder_output_size != model_config.hidden_size:
                self.linear_in = torch.nn.Linear(
                    encoder_output_size, model_config.hidden_size
                )
            else:
                self.linear_in = torch.nn.Identity()

            self.tokenizer_padding_side = "right"
            self.decoder_word_embeddings = self.decoder.plm.transformer.word_embeddings

            self.prefix = torch.nn.Parameter(
                self.decoder_word_embeddings(
                    tokenizer.encode(prefix, return_tensors="pt")
                ).detach(),
                requires_grad=False,
            )
            self.postfix = torch.nn.Parameter(
                self.decoder_word_embeddings(
                    tokenizer.encode(postfix, return_tensors="pt")
                ).detach(),
                requires_grad=False,
            )

            self.prefix_tuning = True
        else:
            self.prefix_tuning = False

            self.causal_lm = causal_lm

            self._init_args = {
                "vocab_size": vocab_size,
                "encoder_output_size": encoder_output_size,
                "model_name_or_path": model_name_or_path,
                "prefix": prefix,
                "postfix": postfix,
                "load_in_8bit": load_in_8bit,
                "torch_compile": torch_compile,
                "map_device": torch.device("cpu"),
            }

            self.linear_in = torch.nn.Identity()
            # self.linear_in = torch.nn.Linear(1024, 4096)

            if load_in_8bit:
                # self._init()
                self.decoder = None
            else:
                self._init()

    def forward(
        self,
        hs_pad: torch.Tensor,
        hlens: torch.Tensor,
        ys_in_pad: torch.Tensor,
        ys_in_lens: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward decoder.

        Args:
            hs_pad: encoded memory, float32  (batch, maxlen_in, feat)
            hlens: (batch)
            ys_in_pad: input tensor (batch, maxlen_out, #mels)
            ys_in_lens: (batch)
        Returns:
            (tuple): tuple containing:

            x: decoded token score before softmax (batch, maxlen_out, token)
                if use_output_layer is True,
            olens: (batch, )
        """
        if self.decoder is None:
            self._init_args["map_device"] = hs_pad.device
            self._init()

        enc_out = self.linear_in(hs_pad)

        if self.causal_lm:
            prefix, prefix_lengths, postfix, postfix_lengths = None, None, None, None

            if "prefix" in kwargs:
                prefix = kwargs["prefix"]
                prefix_lengths = kwargs["prefix_lengths"]

            if "postfix" in kwargs:
                postfix = kwargs["postfix"]
                postfix_lengths = kwargs["postfix_lengths"]

            args, no_loss_lengths = self.add_prefix_postfix(
                enc_out,
                hlens,
                ys_in_pad,
                ys_in_lens,
                prefix,
                prefix_lengths,
                postfix,
                postfix_lengths,
            )
        else:
            args = {"return_dict": True}

            if self.decoder.__class__.__name__ == "MBartDecoder":
                ys_in_pad[:, 0] = 2

            args["input_ids"] = ys_in_pad
            mask = (~make_pad_mask(ys_in_lens)).to(ys_in_pad.device).float()
            args["attention_mask"] = mask

            args["encoder_hidden_states"] = enc_out
            hs_mask = (~make_pad_mask(hlens)).to(hs_pad.device).float()
            args["encoder_attention_mask"] = hs_mask

        if self.prefix_tuning:
            x = self.decoder.prompt_model(args).logits
        else:
            x = self.decoder(**args).last_hidden_state

        if self.causal_lm:
            if self.tokenizer_padding_side == "left":
                x = torch.vstack(
                    [
                        F.pad(
                            x[i, -ys_in_lens[i] :, :],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )
            else:
                x = torch.vstack(
                    [
                        F.pad(
                            x[
                                i,
                                no_loss_lengths[i] : no_loss_lengths[i] + ys_in_lens[i],
                                :,
                            ],
                            (0, 0, 0, ys_in_lens.max() - ys_in_lens[i]),
                        ).unsqueeze(0)
                        for i in range(x.shape[0])
                    ]
                )

        if not self.prefix_tuning:
            x = self.lm_head(x)

        return x, ys_in_lens

    def reload_pretrained_parameters(self):
        # self.decoder.load_state_dict(self.decoder_pretrained_params)

        # if self.lm_head_pretrained_params is not None:
        #    self.lm_head.load_state_dict(self.lm_head_pretrained_params)

        logging.info("Pretrained Transformers model parameters reloaded!")

    def _init(self):
        vocab_size = self._init_args["vocab_size"]
        model_name_or_path = self._init_args["model_name_or_path"]
        prefix = self._init_args["prefix"]
        postfix = self._init_args["postfix"]
        load_in_8bit = self._init_args["load_in_8bit"]

        if self.causal_lm:
            if load_in_8bit:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map={
                        "transformer": self._init_args["map_device"].index,
                        "lm_head": self._init_args["map_device"].index,
                        "word_embeddings": self._init_args["map_device"].index,
                        "word_embeddings_layernorm": self._init_args[
                            "map_device"
                        ].index,
                        "h": self._init_args["map_device"].index,
                        "ln_f": self._init_args["map_device"].index,
                    },
                    load_in_8bit=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
            self.decoder = model.transformer
            self.decoder_pad_token_id = self.decoder.config.pad_token_id
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)

            if hasattr(model, "model"):
                self.decoder = model.model.decoder
            else:
                self.decoder = model.decoder

        model.resize_token_embeddings(vocab_size)

        self.lm_head = model.lm_head
        self.model_name_or_path = model_name_or_path

        self.linear_in = self.linear_in.to(self._init_args["map_device"])
        # if encoder_output_size != self.decoder.config.hidden_size:
        #    self.linear_in = torch.nn.Linear(
        #        encoder_output_size, self.decoder.config.hidden_size
        #    ).to(self._init_args["map_device"])
        # else:
        #    self.linear_in = torch.nn.Identity()

        if model.is_loaded_in_8bit:
            model = prepare_model_for_int8_training(model)
            model.config.use_cache = False

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.tokenizer_padding_side = tokenizer.padding_side

        self.prefix = torch.nn.Parameter(
            self.decoder.word_embeddings(
                tokenizer.encode(prefix, return_tensors="pt").to(
                    device=self._init_args["map_device"]
                )
            ).detach(),
            requires_grad=False,
        )
        self.postfix_tokens = tokenizer.encode(postfix, return_tensors="pt").to(
            device=self._init_args["map_device"]
        )
        self.postfix = torch.nn.Parameter(
            self.decoder.word_embeddings(self.postfix_tokens).detach(),
            requires_grad=False,
        )

        if self._init_args["torch_compile"]:
            self.decoder = torch.compile(self.decoder)

    def add_prefix_postfix(
        self,
        enc_out,
        hlens,
        ys_in_pad,
        ys_in_lens,
        prefix=None,
        prefix_lengths=None,
        postfix=None,
        postfix_lengths=None,
    ):
        args = {}

        if prefix is not None:
            ys_prefixes = self.decoder.word_embeddings(prefix.long())
        else:
            ys_prefixes = self.prefix.repeat(ys_in_pad.shape[0], 1, 1)
            prefix_lengths = (
                torch.tensor(self.prefix.shape[1])
                .repeat(ys_in_pad.shape[0])
                .to(enc_out.device)
            )

        if postfix is not None:
            ys_postfixes = self.decoder.word_embeddings(postfix.long())
        else:
            ys_postfixes = self.postfix.repeat(ys_in_pad.shape[0], 1, 1)
            postfix_lengths = (
                torch.tensor(self.postfix.shape[1])
                .repeat(ys_in_pad.shape[0])
                .to(enc_out.device)
            )

        no_loss_lengths = (
            prefix_lengths + hlens + postfix_lengths - 1
        )  # the last element of postfix should predict ys_in_lens[1]
        inputs_lengths = no_loss_lengths + ys_in_lens  # ys_in_lens[0] is <sos>

        enc_out_list = []

        for i in range(len(hlens)):
            enc_out_element = [
                ys_prefixes[i : i + 1, : prefix_lengths[i]],
                enc_out[i : i + 1, : hlens[i], :],
                ys_postfixes[i : i + 1, : postfix_lengths[i]],
                self.decoder.word_embeddings(ys_in_pad[i : i + 1, 1 : ys_in_lens[i]]),
            ]

            padding = self.decoder.word_embeddings(
                torch.tensor([[self.decoder_pad_token_id]]).to(enc_out.device)
            ).expand(
                -1,
                inputs_lengths.max()
                - (
                    prefix_lengths[i]
                    + hlens[i]
                    + postfix_lengths[i]
                    + ys_in_lens[i]
                    - 1
                ),
                -1,
            )

            if self.tokenizer_padding_side == "left":
                enc_out_element.insert(0, padding)
            else:
                enc_out_element.insert(len(enc_out_element), padding)

            enc_out_list.append(torch.cat(enc_out_element, dim=1))

        args["inputs_embeds"] = torch.vstack(enc_out_list)
        if self._init_args["load_in_8bit"]:
            args["inputs_embeds"] = args["inputs_embeds"].to(torch.half)

        hs_mask = (~make_pad_mask(inputs_lengths)).to(enc_out.device).float()

        if self.tokenizer_padding_side == "left":
            args["attention_mask"] = hs_mask.flip([1])
        else:
            args["attention_mask"] = hs_mask

        if not self.prefix_tuning:
            args["return_dict"] = True

        return args, no_loss_lengths
