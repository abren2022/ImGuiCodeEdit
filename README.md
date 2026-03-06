(ad_env) user@user-System-Product-Name:~/projects/stable-diffusion-webui$ ./webui.sh --xformers
/home/user/projects/stable-diffusion-webui/webui-user.sh: line 20: sexport: command not found

################################################################
Install script for stable-diffusion + Web UI
Tested on Debian 11 (Bullseye), Fedora 34+ and openSUSE Leap 15.4 or newer.
################################################################

################################################################
Running on user user
################################################################

################################################################
Repo already cloned, using it as install directory
################################################################

################################################################
python venv already activate or run without venv: 
################################################################

################################################################
Launching launch.py...
################################################################
glibc version is 2.35
Cannot locate TCMalloc. Do you have tcmalloc or google-perftool installed on your system? (improves CPU memory usage)
fatal: not a git repository (or any of the parent directories): .git
fatal: not a git repository (or any of the parent directories): .git
Python 3.10.15 (main, Oct  3 2024, 07:27:34) [GCC 11.2.0]
Version: 1.10.1
Commit hash: <none>
Cloning assets into /home/user/projects/stable-diffusion-webui/repositories/stable-diffusion-webui-assets...
Cloning into '/home/user/projects/stable-diffusion-webui/repositories/stable-diffusion-webui-assets'...
fatal: unable to access 'https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git/': Could not resolve host: github.com
Traceback (most recent call last):
  File "/home/user/projects/stable-diffusion-webui/launch.py", line 48, in <module>
    main()
  File "/home/user/projects/stable-diffusion-webui/launch.py", line 39, in main
    prepare_environment()
  File "/home/user/projects/stable-diffusion-webui/modules/launch_utils.py", line 411, in prepare_environment
    git_clone(assets_repo, repo_dir('stable-diffusion-webui-assets'), "assets", assets_commit_hash)
  File "/home/user/projects/stable-diffusion-webui/modules/launch_utils.py", line 192, in git_clone
    run(f'"{git}" clone --config core.filemode=false "{url}" "{dir}"', f"Cloning {name} into {dir}...", f"Couldn't clone {name}", live=True)
  File "/home/user/projects/stable-diffusion-webui/modules/launch_utils.py", line 116, in run
    raise RuntimeError("\n".join(error_bits))
RuntimeError: Couldn't clone assets.
Command: "git" clone --config core.filemode=false "https://github.com/AUTOMATIC1111/stable-diffusion-webui-assets.git" "/home/user/projects/stable-diffusion-webui/repositories/stable-diffusion-webui-assets"
Error code: 128
