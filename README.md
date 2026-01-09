Traceback (most recent call last):
  File "/home/user/projects/mmaction2-main/tools/train.py", line 143, in <module>
    main()
  File "/home/user/projects/mmaction2-main/tools/train.py", line 139, in main
    runner.train()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/runner.py", line 1777, in train
    model = self.train_loop.run()  # type: ignore
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/loops.py", line 98, in run
    self.run_epoch()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/runner/loops.py", line 114, in run_epoch
    for idx, data_batch in enumerate(self.dataloader):
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 630, in __next__
    data = self._next_data()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1344, in _next_data
    return self._process_data(data)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/dataloader.py", line 1370, in _process_data
    data.reraise()
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/_utils.py", line 706, in reraise
    raise exception
AssertionError: Caught AssertionError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/worker.py", line 309, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 410, in __getitem__
    data = self.prepare_data(idx)
  File "/home/user/miniforge3/envs/ActionRecognition/lib/python3.10/site-packages/mmengine/dataset/base_dataset.py", line 792, in prepare_data
    data_info = self.get_data_info(idx)
  File "/home/user/projects/mmaction2-main/mmaction/datasets/ava_dataset.py", line 323, in get_data_info
    data_info['proposals'].min() >= 0, \
AssertionError: relative proposals invalid: max value 0.9975249791666667, min value -1.9999999999881224e-07
