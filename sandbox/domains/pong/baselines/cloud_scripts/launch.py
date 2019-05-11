import os

def make_instance(snapshot_name, instance_name):
    ret = \
    f"""
gcloud compute --project "capgroup" disks create "{instance_name}" --size "100" --zone "us-east1-b" --source-snapshot "{snapshot_name}" --type "pd-standard"

gcloud compute --project=capgroup instances create {instance_name} --zone=us-east1-b --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE --service-account=881379218043-compute@developer.gserviceaccount.com --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append --accelerator=type=nvidia-tesla-p100,count=1 --tags=http-server,https-server --disk=name=instance-1,device-name=instance-1,mode=rw,boot=yes,auto-delete=yes

    """
    return ret

def run_remote_cmd(instance_name, folder_path, script_path):
    """
    instance_name : instance-1
    folder_path : home/yewenpu/embedded_prototype/
    script_path : ./scripts/run.sh
    """
    ret = \
    f"""
gcloud compute ssh --zone=us-east1-b yewenpu@{instance_name} --command='source ~/.bashrc; cd {folder_path}; git checkout {script_path}; git pull; touch lalalala; chmod 777 {script_path}; {script_path}'
    """
    return ret

def start_up_remove(instance_name):
    return f"""
gcloud compute instances start {instance_name}

    """

def shut_off_remote(instance_name):
    return f"""
gcloud compute ssh --zone=us-east1-b yewenpu@{instance_name} --command='sudo poweroff'
    """


if __name__ == "__main__":
    # snapshot_name = "snapshot-test-pytorch1"
    instance_name = "instance-beef1"
    # print (make_instance(snapshot_name, instance_name))

    folder_path = "/home/yewenpu/embedded_prototype/sandbox/domains/pong/baselines"
    script_path = "./cloud_scripts/run.sh" 
    run_experiment = run_remote_cmd(instance_name, folder_path, script_path)
    os.system(run_experiment)
