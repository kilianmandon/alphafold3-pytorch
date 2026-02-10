import os
from pathlib import Path
import subprocess
import time
from dotenv import load_dotenv
from vastai_sdk import VastAI

def select_offer(vast_sdk, num_gpus=4, top_k=20):
    offers = vast_sdk.search_offers(query=f'num_gpus={num_gpus} rented=False rentable=True')
    offers = offers[:top_k]

    graphics_cards = [v["gpu_name"] for v in offers]
    max_len_graphiucs_cards = max([len(g) for g in graphics_cards])
    prices = [f'{v["dph_total"]:.2f} $' for v in offers]
    max_len_prices = max([len(p) for p in prices])
    gpu_rams = [f'{v["gpu_ram"]/1024:.1f} GB' for v in offers]
    max_len_gpu_rams = max([len(r) for r in gpu_rams])
    geolocations = [v["geolocation"] for v in offers]
    max_len_geolocations = max([len(g) for g in geolocations])

    inet_up = [f'{v["inet_up"]/1000:.1f} Gbps' for v in offers]
    max_len_inet_up = max([len(i) for i in inet_up])
    inet_down = [f'{v["inet_down"]/1000:.1f} Gbps' for v in offers]
    max_len_inet_down = max([len(i) for i in inet_down])

    for i, v in enumerate(offers):
        idx_str = f'({i+1})'
        print(f'{idx_str:>4} {v["gpu_name"]:{max_len_graphiucs_cards}} - {prices[i]:>{max_len_prices}} - {gpu_rams[i]:>{max_len_gpu_rams}} - {inet_up[i]:>{max_len_inet_up}} - {inet_down[i]:>{max_len_inet_down}} - {geolocations[i]:<{max_len_geolocations}}')

    offer_idx = int(input('Select offer: ')) - 1
    return offers[offer_idx]

def wait_for_online(vast_sdk, instance_id, timeout=300):
    warmup_time = 30
    start_time = time.time()
    print('Loading...', end='')
    while time.time() - start_time < timeout:
        data = vast_sdk.show_instance(id=instance_id)
        if data['actual_status'] in ['loading', 'created']:
            print('.', end='', flush=True)
            time.sleep(3)
        elif data['actual_status'] == 'running':
            print('')
            return True
        elif time.time() - start_time > warmup_time:
            print('')
            print(f'Unexpected status: {data["actual_status"]}')
            return False

def ensure_ssh_master(instance_data):
    control_path = f"/tmp/vastai_ssh_{instance_data['id']}"

    # Check if master is already running
    check_cmd = f'ssh -O check -o ControlPath={control_path} -o StrictHostKeyChecking=no root@{instance_data["ssh_host"]} 2>/dev/null'
    result = subprocess.run(check_cmd, shell=True, capture_output=True)
    
    if result.returncode == 0:
        return  # Master already running
    
    ssh_identify_file = '~/.ssh/id_vastai'

    ssh_opts = (
        f"-i {ssh_identify_file} "
        f"-p {instance_data['ssh_port']} "
        f"-o ControlMaster=yes "
        f"-o ControlPersist=10m "
        f"-o ControlPath={control_path} "
        f"-o StrictHostKeyChecking=no"
    )
    cmd = f'ssh {ssh_opts} -fN root@{instance_data["ssh_host"]}'
    timeout = 60
    start_time = time.time()
    retry_started = False
    while time.time() - start_time < timeout:
        try:
            subprocess.run(cmd, shell=True, check=True)
            if retry_started:
                print('')
            return
        except subprocess.CalledProcessError as e:
            if not retry_started:
                print(f"SSH master setup failed: {e}. Retrying...", end='')
                retry_started = True
            else:
                print('.', end='', flush=True)
            time.sleep(5)
            continue

def upload_file(source, destination, instance_data, vast_sdk):
    ensure_ssh_master(instance_data)
    dst_path = f'root@{instance_data["ssh_host"]}:{destination}'

    control_path = f"/tmp/vastai_ssh_{instance_data['id']}"
    ssh_opts = f"-o ControlPath={control_path} -o LogLevel=ERROR"

    mkdir_cmd = f'ssh {ssh_opts} root@{instance_data["ssh_host"]} "mkdir -p {Path(destination).parent}"'
    cmd = f'rsync -avz -e "ssh {ssh_opts}" {source} {dst_path}'

    subprocess.run(mkdir_cmd, shell=True, check=True)
    subprocess.run(cmd, shell=True, check=True)


def execute_remote_command(instance_data, command):
    ensure_ssh_master(instance_data)
    control_path = f"/tmp/vastai_ssh_{instance_data['id']}"
    ssh_opts = f"-o ControlPath={control_path} -o LogLevel=QUIET"
    
    cmd = f'ssh {ssh_opts} root@{instance_data["ssh_host"]} "{command}"'
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    return result.stdout
        

def main():
    load_dotenv()
    vast_sdk = VastAI(api_key=os.getenv("VAST_API_KEY"))

    offer = select_offer(vast_sdk)
    instance = vast_sdk.create_instance(id=offer["id"], image="vastai/base-image:@vastai-automatic-tag", disk=100)
    if instance == '' or not instance["success"]:
        print("Failed to create instance")
        return

    instance_id = instance["new_contract"]

    if not wait_for_online(vast_sdk, instance_id):
        print("Instance failed to come online")
        return

    instance_data = vast_sdk.show_instance(id=instance_id)

    print('Instance created, uploading files...')
    config_name = 'config_lr_5e-4.yaml'
    upload_file(f'data/configs/{config_name}', f'/workspace/{config_name}', instance_data, vast_sdk)
    upload_file('cloud_infrastructure/vastai/startup_script.sh', '/workspace/startup_script.sh', instance_data, vast_sdk)

    print('Files uploaded, starting training.')

    cmd = f"""
apt install -y screen && screen -dmS train bash -c '
export WANDB_API_KEY="{os.environ['WANDB_API_KEY']}"
export AWS_ACCESS_KEY_ID="{os.environ['AWS_ACCESS_KEY_ID']}"
export AWS_SECRET_ACCESS_KEY="{os.environ['AWS_SECRET_ACCESS_KEY']}"
export AWS_REGION="{os.environ['AWS_REGION']}"
export VAST_API_KEY="{os.environ['VAST_API_KEY']}"
export instance_id="{instance_id}"
export config_path="/workspace/{config_name}"
bash /workspace/startup_script.sh
'
"""

    print(execute_remote_command(instance_data, cmd))
    ...



if __name__ == "__main__":
    main()