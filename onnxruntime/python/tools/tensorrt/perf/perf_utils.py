import subprocess
import json
import pprint
import logging
import coloredlogs
import re
import sys

debug = False
debug_verbose = False 

def get_output(command):
    p = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    return p.stdout.decode("ascii").strip()

def find(regex_string): 
    import glob
    results = glob.glob(regex_string)
    results.sort()
    return results

def pretty_print(pp, json_object):
    pp.pprint(json_object)
    sys.stdout.flush()

def get_latest_commit_hash():
    return get_output(["git", "rev-parse", "--short", "HEAD"])

def parse_single_file(f):

    try:
        data = json.load(f)
    except Exception as e:
        return None

    model_run_flag = False
    first_run_flag = True
    provider_op_map = {}  # ep -> map of operator to duration
    provider_op_map_first_run = {} # ep -> map of operator to duration

    for row in data:
        if "cat" not in row:
            continue

        if row["cat"] == "Session":
            if "name" in row and row["name"] == "model_run":
                if not first_run_flag:
                    break

                model_run_flag = True
                first_run_flag = False

        elif row["cat"] == "Node":
            if "name" in row and "args" in row and re.search(".*kernel_time", row["name"]):
                args = row["args"]

                if "op_name" not in args or "provider" not in args:
                    continue

                provider = args["provider"]

                if first_run_flag:
                    if provider not in provider_op_map_first_run:
                        provider_op_map_first_run[provider] = {}

                    op_map = provider_op_map_first_run[provider]

                    if row["name"] in op_map:
                        provider_op_map[provider] = {}
                        op_map = provider_op_map[provider]
                        op_map[row["name"]] = row["dur"]
                        provider_op_map[provider] = op_map
                    else:
                        op_map[row["name"]] = row["dur"]
                        provider_op_map_first_run[provider] = op_map
                else:
                    if provider not in provider_op_map:
                        provider_op_map[provider] = {}

                    op_map = provider_op_map[provider]

                    # avoid duplicated metrics
                    if row["name"] not in op_map:
                        op_map[row["name"]] = row["dur"]
                        provider_op_map[provider] = op_map


    if debug_verbose:
        pprint._sorted = lambda x:x
        pprint.sorted = lambda x, key=None: x
        pp = pprint.PrettyPrinter(indent=4)
        print("------First run ops map (START)------")
        for key, map in provider_op_map_first_run.items():
            print(key)
            pp.pprint(dict(sorted(map.items(), key=lambda item: item[1], reverse=True)))

        print("------First run ops map (END) ------")
        print("------Second run ops map (START)------")
        for key, map in provider_op_map.items():
            print(key)
            pp.pprint(dict(sorted(map.items(), key=lambda item: item[1], reverse=True)))
        print("------Second run ops map (END) ------")

    return provider_op_map if model_run_flag else None

def calculate_cuda_op_percentage(cuda_op_map):
    if not cuda_op_map or len(cuda_op_map) == 0:
        return 0

    cuda_ops = 0
    cpu_ops = 0
    for key, value in cuda_op_map.items():
        if key == 'CPUExecutionProvider':
            cpu_ops += len(value)

        elif key == 'CUDAExecutionProvider':
            cuda_ops += len(value)

    return cuda_ops / (cuda_ops + cpu_ops)

##########################################
# Return: total ops executed in TRT,
#         total ops,
#         ratio of ops executed in TRT,
##########################################
def calculate_trt_op_percentage(trt_op_map, cuda_op_map):
    # % of TRT ops
    total_ops = 0
    total_cuda_and_cpu_ops = 0
    for ep in ["CUDAExecutionProvider", "CPUExecutionProvider"]:
        if ep in cuda_op_map:
            op_map = cuda_op_map[ep]
            total_ops += len(op_map)

        if ep in trt_op_map:
            op_map = trt_op_map[ep]
            total_cuda_and_cpu_ops += len(op_map)

    if total_ops == 0:
        print("Error ...")
        raise

    if len(trt_op_map) == 0:
        total_cuda_and_cpu_ops = total_ops

    #
    # equation of % TRT ops:
    # (total ops in cuda json - cuda and cpu ops in trt json)/ total ops in cuda json
    #
    ratio_of_ops_in_trt = (total_ops - total_cuda_and_cpu_ops) / total_ops
    if debug:
        print(f"total_cuda_and_cpu_ops: {total_cuda_and_cpu_ops}")
        print(f"total_ops: {total_ops}")
        print(f"ratio_of_ops_in_trt: {ratio_of_ops_in_trt}")

    return ((total_ops - total_cuda_and_cpu_ops), total_ops, ratio_of_ops_in_trt)

def get_total_ops(op_map):
    return sum(
        len(op_map[ep])
        for ep in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ep in op_map
    )


##########################################
# Return: total TRT execution time,
#         total execution time,
#         ratio of execution time in TRT
##########################################
def calculate_trt_latency_percentage(trt_op_map):
    # % of TRT execution time
    total_execution_time = 0
    total_trt_execution_time = 0
    for ep in ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]:
        if ep in trt_op_map:
            op_map = trt_op_map[ep]

            total_time = sum(int(value) for key, value in op_map.items())
            if ep == "TensorrtExecutionProvider":
                total_trt_execution_time = total_time

            total_execution_time += total_time



    if total_execution_time == 0:
        ratio_of_trt_execution_time = 0
    else:
        ratio_of_trt_execution_time = total_trt_execution_time / total_execution_time

    if debug:
        print(f"total_trt_execution_time: {total_trt_execution_time}")
        print(f"total_execution_time: {total_execution_time}")
        print(f"ratio_of_trt_execution_time: {ratio_of_trt_execution_time}")

    return (total_trt_execution_time, total_execution_time, ratio_of_trt_execution_time)



def get_profile_metrics(path, profile_already_parsed, logger=None):
    logger.info(f"Parsing/Analyzing profiling files in {path} ...")
    p1 = subprocess.Popen(["find", path, "-name", "onnxruntime_profile*", "-printf", "%T+\t%p\n"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["sort"], stdin=p1.stdout, stdout=subprocess.PIPE)
    stdout, sterr = p2.communicate()
    stdout = stdout.decode("ascii").strip()
    profiling_files = stdout.split("\n")
    logger.info(profiling_files)

    data = []
    for profile in profiling_files:
        profile = profile.split('\t')[1]
        if profile in profile_already_parsed:
            continue
        profile_already_parsed.add(profile)

        logger.info(f"start to parse {profile} ...")
        with open(profile) as f:
            if op_map := parse_single_file(f):
                data.append(op_map)

    if not data:
        logger.info("No profile metrics got.")
        return None

    return data[-1]

