#!/usr/bin/env python 
import argparse
import ncluster
import os

IMAGE_NAME = 'pytorch_openmpi_source'
INSTANCE_TYPE = 'c5.xlarge'

ncluster.set_backend('aws')
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='imagenet',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=24,
                    help="how many machines to use")
args = parser.parse_args()


def main():
  supported_regions = ['us-west-2', 'us-east-1', 'us-east-2']
  assert ncluster.get_region() in supported_regions, f"required AMI {IMAGE_NAME} has only been made available in regions {supported_regions}, but your current region is {ncluster.get_region()}"
  # assert args.machines in schedules, f"{args.machines} not supported, only support {schedules.keys()}"

  os.environ['NCLUSTER_AWS_FAST_ROOTDISK'] = '1'  # use io2 disk on AWS
  job = ncluster.make_job(name=args.name,
                          run_name=f"{args.name}-{args.machines}",
                          num_tasks=args.machines,
                          image_name=IMAGE_NAME,
                          instance_type=INSTANCE_TYPE,
                          install_script=open('setup.sh').read())
  job.upload('distributed_experiments')
  #job.run(f'source activate pytorch_source')

  # nccl_params = get_nccl_params(args.machines, NUM_GPUS)

  # Training script args
  # default_params = [
    # '~/data/imagenet',
    # '--fp16',
    # '--logdir', job.logdir,
    # '--distributed',
    # '--init-bn0',
    # '--no-bn-wd',
  # ]

  # params = ['--phases', schedules[args.machines]]
  # training_params = default_params + params
  # training_params = ' '.join(map(format_params, training_params))
  master_addr = "'tcp://{}:2345'".format(job.tasks[0].ip)
  gloo_name = "'gloo'"
  # TODO: simplify args processing, or give link to actual commands run
  for i, task in enumerate(job.tasks):
    dist_params = f'-a resnet50 --lr 0.01  --dist-url {master_addr} --dist-backend {gloo_name} --world-size 24 --rank {i} -sub 32 ~/data'
    cmd = f'python distributed_experiments/imagenet_example_cpu.py {dist_params}'
    # task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
    task.run(cmd, non_blocking=True)

  print(f"Logging to {job.logdir}")


if __name__ == '__main__':
  main()
