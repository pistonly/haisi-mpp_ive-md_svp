import telnetlib
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

def do_telnet(tn, finish, commands):
    for command in commands:
        tn.write(command.encode('ascii') + b'\n')
        time.sleep(2)

    tn.write(b'\n')
    tn.read_until(finish.encode('ascii'))

def link_ss928(Host, username, password, finish):
    try:
        tn = telnetlib.Telnet(Host, port=23, timeout=10)
        tn.set_debuglevel(2)

        tn.read_until(b'localhost login: ')
        tn.write(username.encode('ascii') + b'\n')

        tn.read_until(b'Password: ')
        tn.write(password.encode('ascii') + b'\n')

        tn.read_until(finish.encode('ascii'))

        return tn
    except Exception as e:
        print(f"Failed to connect {Host}: {e}")
        return None

if __name__ == '__main__':
    ss928_ip = ['11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, nargs="+")
    args = parser.parse_args()
    if args.ip:
        ss928_ip = args.ip

    Host = '192.168.0.'
    username = 'root'
    password = ''
    finish = '~ #'
    target_dir = "/mnt/data/mpp_ive-md_svp"
    mount_dir = "/root/win"
    mount_work_dir = f"{mount_dir}/deploy/mpp_ive-md_svp"

    commands = [
        f'mkdir -p {mount_dir} && mount -t nfs -o nolock 192.168.0.111:/d/nfs_share {mount_dir}',
        f'''nohup sh -c "cp {mount_work_dir}/data/*.json {target_dir}/data/ && reboot" > /root/copyfile.log 2>&1 &''',
    ]

    def execute_on_device(ip):
        tn = link_ss928(Host + ip, username, password, finish)
        if tn:
            do_telnet(tn, finish, commands)
            tn.close()

    with ThreadPoolExecutor(max_workers=6) as executor:
        executor.map(execute_on_device, ss928_ip)
