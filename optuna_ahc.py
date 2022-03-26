import optuna
import subprocess


def objective(trial):
    score = 0
    start_temp = trial.suggest_float('start_temp', 50, 200)
    end_temp = trial.suggest_float('end_temp', 0.01, 50)
    if start_temp < end_temp:
        start_temp, end_temp = end_temp, start_temp
    LOOP_TIME = 10
    for i in range(LOOP_TIME):
        with open(f"./tools/in/{i:04}.txt", "a") as f:
            f.write(f"{start_temp}\n")
            f.write(f"{end_temp}")
        with open(f"./tools/in/{i:04}.txt", "r") as f:
            res = subprocess.run(["cargo",
                                  "run",
                                  "--quiet",
                                  "--release",
                                  "--bin",
                                  "ahc009-a"],
                                 stdin=f,
                                 capture_output=True,
                                 text=True)
            out = res.stdout.split()
            score += float(out[0]) if out else 0
        subprocess.run(["sed", "-i", "$d", f"./tools/in/{i:04}.txt"])
        subprocess.run(["sed", "-i", "$d", f"./tools/in/{i:04}.txt"])
    # print('start temp: %1.3f,end temp: %1.3f score: %1.3f' %
    #       (start_temp, end_temp, score / 5.0))
    return score


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(study.best_params)
print(study.best_value)
