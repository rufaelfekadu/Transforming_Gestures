import time
import os
import logging

logger = logging.getLogger(__name__)

timelimit_argument_str = "-t 3-23:00:00"
# timelimit_argument_str = "-t 2-23:00:00"
GPU_argument_str = "--gres=gpu:1"
CPU_argument_str = "-c 1"
DEFAULT_MEM = 20000
# RAM_argument_str = "--mem 20000"

GPU_partition_argument_str = "-p power-general gpu-general"
CPU_partition_argument_str = "-p power-general"

# CPU_exclude_nodes_str = "--exclude=ielsc-62,ielsc-65,ielsc-68,ielsc-75,ielsc-84,ielsc-85,ielsc-110,ielsc-111,ielsc-112,ielsc-113,ielsc-114,ielsc-115,ielsc-116,ielsc-117"
CPU_exclude_nodes_str = ""
# node_list = list(range(100, 108)) + list(range(110,113)) + list(range(114,118)) + list(range(54, 58))
# CPU_exclude_nodes_str = "--nodelist="
# for i, node in enumerate(node_list):
#     if i != 0:
#         CPU_exclude_nodes_str += ","
#     CPU_exclude_nodes_str += "ielsc-{}".format(node)

class SlurmJobState:
    def __init__(self, job, state, extra):
        self.job = job
        self.state = state
        self.extra = extra

    def __repr__(self):
        return "SlurmJobState(job={}({}), state={})"\
        .format(self.job.job_name, self.job.job_id, self.state)

    def is_successfull(self):
        return self.state.startswith("COMPLETED")


class SlurmJob:
    def __init__(self, job_name, job_folder, run_line, run_on_GPU=False, timelimit=False, mem=DEFAULT_MEM):
        self.job_name = job_name
        self.job_filename = os.path.join(job_folder, job_name + ".job")
        self.log_filename = os.path.join(job_folder, job_name + ".log")
        self.run_on_GPU = run_on_GPU
        self.run_line = run_line
        self.job_id = None
        self.timelimit = timelimit
        self.mem = mem

    def send(self):
        # write a job file and run it
        with open(self.job_filename, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --job-name %s\n" %(self.job_name))
            fh.writelines("#SBATCH -o %s\n" %(self.log_filename))
            if self.run_on_GPU:
                fh.writelines("#SBATCH %s\n" %(GPU_partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(GPU_argument_str))
            else:
                fh.writelines("#SBATCH %s\n" %(CPU_partition_argument_str))
                fh.writelines("#SBATCH %s\n" %(CPU_exclude_nodes_str))

            if self.timelimit:
                fh.writelines("#SBATCH %s\n" %(timelimit_argument_str))
            fh.writelines("#SBATCH %s\n" %(CPU_argument_str))
            fh.writelines("#SBATCH --mem %s\n" %(self.mem))
            fh.writelines(self.run_line)

        popen_output = os.popen("/usr/bin/sbatch %s" %(self.job_filename)).read()
        self.job_id = int(popen_output.split(" ")[3][:-1])
        logger.info("Job {}({}) submitted".format(self.job_name, self.job_id))
    
    def join(self, timeout=None):
        total_time_waited = 0
        while True:
            if timeout and total_time_waited > timeout:
                raise Exception("timeout reached") # TODO: better exception type
            try:
                # logger.info("Trying seff {}".format(self.job_id))
                popen_output = os.popen("/usr/bin/seff {}".format(self.job_id)).read()
                # logger.info(popen_output[:-1])
                print(popen_output)
                state_parts = popen_output.split("\n")[3][7:].split(" ")
                # logger.info(state_parts)
                    
                state = state_parts[0]
                extra = ""
                if len(state_parts) > 1:
                    extra = state_parts[1:]
                state_obj = SlurmJobState(self, state, extra)
                # logger.info(state)
                # logger.info(extra)
                
                if state.startswith("COMPLETED"):
                    logger.info("Job {}({}) completed successfully".format(self.job_name, self.job_id))
                    return state_obj
                if state.startswith("FAILED") or state.startswith("CANCELLED"):
                    logger.info("Job {}({}) failed with state {}".format(self.job_name, self.job_id, state))
                    return state_obj
            except:
                pass
            time.sleep(10)
            total_time_waited += 10

    def cancel(self):
        raise NotImplementedError()

class SlurmJobFactoryState:
    def __init__(self, states):
        self.states = states
        self.succesfull_states = [(s,e) for (s,e) in self.states if s.is_successfull()]
        self.unsuccessfull_states = [(s,e) for (s,e) in self.states if not s.is_successfull()]

    def __repr__(self):
        return "SlurmJobFactoryState(number of jobs={}, sucessfull jobs={})"\
            .format(len(self.states), len(self.succesfull_states))

    def is_successfull(self):
        return len(self.states) == len(self.succesfull_states)

class SlurmJobFactory:
    def __init__(self, job_folder):
        self.job_folder = job_folder
        self.jobs = []
        self.old_jobs = []

    def send_job(self, job_name, run_line, run_on_GPU=False, timelimit=False, mem=DEFAULT_MEM, extra=None):
        job = SlurmJob(job_name, self.job_folder, run_line, run_on_GPU, timelimit, mem)
        job.send()
        self.jobs.append((job, extra))

    def join_all(self, on_join=None):
        states = []
        remaining_indexes = list(range(len(self.jobs)))

        to_remove_from_remaining_indexes = []
        while len(remaining_indexes) > 0:
            for index, (job, extra) in enumerate(self.jobs):
                if index in remaining_indexes:
                    try:
                        job_state = job.join(timeout=60)
                        states.append((job_state, extra))
                        if on_join is not None:
                            on_join(job_state, extra)
                        to_remove_from_remaining_indexes.append(index)
                    except:
                        pass

            for index_to_remove in to_remove_from_remaining_indexes:
                remaining_indexes.remove(index_to_remove)
            to_remove_from_remaining_indexes = []

        self.old_jobs += self.jobs
        self.jobs = []

        return SlurmJobFactoryState(states)
