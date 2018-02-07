import re
import subprocess as subp


def nonbusy_gpu():
    def gpu_usage(gpu_id):
        usage, = re.findall(
            r'Gpu\s+: (\d+) %',
            subp.check_output(['nvidia-smi', '--id=' + gpu_id, '-q']).decode())
        return int(usage)

    gpu_ids = re.findall(
        r'GPU (\d+):',
        subp.check_output(['nvidia-smi', '-L']).decode())

    return min(gpu_ids, key=gpu_usage)
