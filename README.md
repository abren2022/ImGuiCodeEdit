Installing collected packages: xformers
  DEPRECATION: Legacy editable install of xformers==0.0.23 from file:///home/user/thirdkt/xformers-0.0.23.post1 (setup.py develop) is deprecated. pip 25.3 will enforce this behaviour change. A possible replacement is to add a pyproject.toml or enable --use-pep517, and use setuptools >= 64. If the resulting installation is not behaving as expected, try using --config-settings editable_mode=compat. Please consult the setuptools documentation for more information. Discussion can be found at https://github.com/pypa/pip/issues/11457
  Running setup.py develop for xformers
    error: subprocess-exited-with-error
    
    × python setup.py develop did not run successfully.
    │ exit code: 1
    ╰─> [79 lines of output]
        fatal: Not a valid object name HEAD
        /home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/dist.py:759: SetuptoolsDeprecationWarning: License classifiers are deprecated.
        !!
        
                ********************************************************************************
                Please consider removing the following classifiers in favor of a SPDX license expression:
        
                License :: OSI Approved :: BSD License
        
                See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
                ********************************************************************************
        
        !!
          self._finalize_license_expression()
        running develop
        /home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/cmd.py:90: DevelopDeprecationWarning: develop command is deprecated.
        !!
        
                ********************************************************************************
                Please avoid running ``setup.py`` and ``develop``.
                Instead, use standards-based tools like pip or uv.
        
                This deprecation is overdue, please update your project and remove deprecated
                calls to avoid build errors in the future.
        
                See https://github.com/pypa/setuptools/issues/917 for details.
                ********************************************************************************
        
        !!
          self.initialize_options()
        Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
        Obtaining file:///home/user/thirdkt/xformers-0.0.23.post1
          Installing build dependencies: started
          Installing build dependencies: finished with status 'error'
          error: subprocess-exited-with-error
        
          × pip subprocess to install build dependencies did not run successfully.
          │ exit code: 1
          ╰─> [8 lines of output]
              Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
              WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def18d0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
              WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1c00>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
              WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1db0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
              WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1f60>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
              WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def2110>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
              ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0 (from versions: none)
              ERROR: No matching distribution found for setuptools>=40.8.0
              [end of output]
        
          note: This error originates from a subprocess, and is likely not a problem with pip.
        error: subprocess-exited-with-error
        
        × pip subprocess to install build dependencies did not run successfully.
        │ exit code: 1
        ╰─> See above for output.
        
        note: This error originates from a subprocess, and is likely not a problem with pip.
        Traceback (most recent call last):
          File "<string>", line 2, in <module>
          File "<pip-setuptools-caller>", line 35, in <module>
          File "/home/user/thirdkt/xformers-0.0.23.post1/setup.py", line 399, in <module>
            setuptools.setup(
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/__init__.py", line 115, in setup
            return distutils.core.setup(**attrs)
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
            return run_commands(dist)
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
            dist.run_commands()
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
            self.run_command(cmd)
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
            super().run_command(command)
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
            cmd_obj.run()
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/command/develop.py", line 39, in run
            subprocess.check_call(cmd)
          File "/home/user/miniforge3/envs/ad_env/lib/python3.10/subprocess.py", line 369, in check_call
            raise CalledProcessError(retcode, cmd)
        subprocess.CalledProcessError: Command '['/home/user/miniforge3/envs/ad_env/bin/python3.10', '-m', 'pip', 'install', '-e', '.', '--use-pep517', '--no-deps']' returned non-zero exit status 1.
        [end of output]
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× python setup.py develop did not run successfully.
│ exit code: 1
╰─> [79 lines of output]
    fatal: Not a valid object name HEAD
    /home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/dist.py:759: SetuptoolsDeprecationWarning: License classifiers are deprecated.
    !!
    
            ********************************************************************************
            Please consider removing the following classifiers in favor of a SPDX license expression:
    
            License :: OSI Approved :: BSD License
    
            See https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#license for details.
            ********************************************************************************
    
    !!
      self._finalize_license_expression()
    running develop
    /home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/cmd.py:90: DevelopDeprecationWarning: develop command is deprecated.
    !!
    
            ********************************************************************************
            Please avoid running ``setup.py`` and ``develop``.
            Instead, use standards-based tools like pip or uv.
    
            This deprecation is overdue, please update your project and remove deprecated
            calls to avoid build errors in the future.
    
            See https://github.com/pypa/setuptools/issues/917 for details.
            ********************************************************************************
    
    !!
      self.initialize_options()
    Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
    Obtaining file:///home/user/thirdkt/xformers-0.0.23.post1
      Installing build dependencies: started
      Installing build dependencies: finished with status 'error'
      error: subprocess-exited-with-error
    
      × pip subprocess to install build dependencies did not run successfully.
      │ exit code: 1
      ╰─> [8 lines of output]
          Looking in indexes: https://pypi.tuna.tsinghua.edu.cn/simple
          WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def18d0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
          WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1c00>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
          WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1db0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
          WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def1f60>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
          WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x70c51def2110>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /simple/setuptools/
          ERROR: Could not find a version that satisfies the requirement setuptools>=40.8.0 (from versions: none)
          ERROR: No matching distribution found for setuptools>=40.8.0
          [end of output]
    
      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: subprocess-exited-with-error
    
    × pip subprocess to install build dependencies did not run successfully.
    │ exit code: 1
    ╰─> See above for output.
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    Traceback (most recent call last):
      File "<string>", line 2, in <module>
      File "<pip-setuptools-caller>", line 35, in <module>
      File "/home/user/thirdkt/xformers-0.0.23.post1/setup.py", line 399, in <module>
        setuptools.setup(
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/__init__.py", line 115, in setup
        return distutils.core.setup(**attrs)
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 186, in setup
        return run_commands(dist)
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 202, in run_commands
        dist.run_commands()
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1002, in run_commands
        self.run_command(cmd)
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/dist.py", line 1102, in run_command
        super().run_command(command)
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 1021, in run_command
        cmd_obj.run()
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/site-packages/setuptools/command/develop.py", line 39, in run
        subprocess.check_call(cmd)
      File "/home/user/miniforge3/envs/ad_env/lib/python3.10/subprocess.py", line 369, in check_call
        raise CalledProcessError(retcode, cmd)
    subprocess.CalledProcessError: Command '['/home/user/miniforge3/envs/ad_env/bin/python3.10', '-m', 'pip', 'install', '-e', '.', '--use-pep517', '--no-deps']' returned non-zero exit status 1.
    [end of output]

note: This error originates from a subprocess, and is likely not a problem with pip.
