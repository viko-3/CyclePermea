# run_with_different_params.py# -*- coding: utf-8 -*-import subprocessimport osimport timeenv = os.environ.copy()env["CUDA_VISIBLE_DEVICES"] = "1"contrastive_learning_margin=[0.05,0.1,0.15,0.2]processes = []# 启动每个参数配置的进程for p1 in contrastive_learning_margin:    print(f"运行程序：param1={p1}")    process = subprocess.Popen(["python", "main.py", "--CL_margin", str(p1)], env=env)    processes.append(process)    time.sleep(60)  # 等待60秒# 等待所有进程完成for process in processes:    process.wait()