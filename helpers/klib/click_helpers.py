from helpers.klib.klib import kdict
import click
import torch
import re
import os

class UnlimitedNargsOption(click.Option):
    '''From https://stackoverflow.com/a/48394004/10917436'''

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(UnlimitedNargsOption, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(UnlimitedNargsOption, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(
                name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval


def process_click_args(ctx: click.Context, cmd_args: dict) -> int:
    cmd_args = kdict(cmd_args)
    if cmd_args.offline:
        os.environ["WANDB_MODE"] = "dryrun"

    ######### Handle CUDA & GPU stuff #########
    num_gpus = len(cmd_args.gpus.split(',')
                   ) if cmd_args.gpus else 0
    if num_gpus > 0:
        # Assume machine has not more than 100 GPUs
        if not re.compile("^([0-9]|[1-9][0-9])(,([0-9]|[1-9][0-9]))*$").fullmatch(cmd_args.gpus):
            ctx.fail(
                f"invalid GPU string specified: \"{cmd_args.gpus}\". Expected format is a comma-seperated list of integers corresponding to GPU indices (execute nvidia-smi for more info)")

        # IMPORTANT: For this to work, the torch import needs to happen afterwards
        os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpus
        import torch
        if not torch.cuda.is_available():
            ctx.fail("GPUs were requested but machine has no CUDA!")
        # torch.cudnn.enabled = True
        # torch.cudnn.benchmark = True
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cmd_args.gpus = num_gpus
    if cmd_args.gpus > 0:
        cmd_args.accelerator = 'ddp'
        # experienced "deadlock" bug with the standard nccl backend
        # os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"
    else:
        cmd_args.accelerator = None
    return cmd_args
