#!/usr/bin/env python3
import argparse
import json
import multiprocessing.pool
import os
import sys
import time

sys.path.append("../mtool")
import main as mtool
sys.path.pop()

def mrp_load(path=None, stream=None):
    assert (path and not stream) or (not path and stream)

    if path:
        stream = open(path, "r", encoding="utf-8")
    graphs, _ = mtool.read_graphs(stream, format="mrp", normalize={"anchors", "case", "edges", "attributes"})
    if path:
        stream.close()

    return graphs

def mrp_score(gold, system, rrhc, mces, parallelize=1):
    assert gold and system

    return mtool.score.mces.evaluate(gold, system, format="json", limits={"rrhc": rrhc, "mces": mces}, cores=parallelize)

    # if parallelize <= 1:
    #     return mtool.score.mces.evaluate(gold, system, format="json", limits={"rrhc": rrhc, "mces": mces})
    # else:
    #     start = time.time()
    #     with multiprocessing.pool.Pool(parallelize) as pool:
    #         scores = pool.starmap(mrp_score, [([g], [s], rrhc, mces) for g, s in mtool.score.core.intersect(gold, system)], 1)

    #     score = {"time": time.time() - start}
    #     if scores:
    #         for k in scores[0]:
    #             if isinstance(scores[0][k], int):
    #                 score[k] = sum(s[k] for s in scores)
    #             elif isinstance(scores[0][k], dict):
    #                 assert set(scores[0][k].keys()) == {"g", "s", "c", "p", "r", "f"}
    #                 score[k] = {i: sum(s[k][i] for s in scores) for i in ["g", "s", "c"]}
    #                 score[k].update({i: v for i, v in zip(["p", "r", "f"], mtool.score.core.fscore(score[k]["g"], score[k]["s"], score[k]["c"]))})
    #             else:
    #                 raise ValueError("Unknown score type {}".format(type(scores[0][k])))

    #     return score

def mrp_tf_log(score, logdir, step, label):
    import tensorflow as tf

    writer = tf.summary.create_file_writer(logdir, filename_suffix=".eval{}.v2".format(step))
    with writer.as_default():
        for k in score:
            if isinstance(score[k], dict):
                tf.summary.scalar("{}/{}".format(label, k), score[k]["f"], step=step)

        for k in score:
            if isinstance(score[k], (int, float)):
                tf.summary.scalar("{}_details/{}".format(label, k), score[k], step=step)
            elif isinstance(score[k], dict):
                for i in score[k]:
                    tf.summary.scalar("{}_details/{}_{}".format(label, k, i), score[k][i], step=step)
    writer.close()

    with open(os.path.join(logdir, label), "a") as eval_file:
        print("Step {}: {}".format(step, ", ".join("{}={}".format(k, score[k]["f"]) for k in score if isinstance(score[k], dict))),
              file=eval_file)

    with open(os.path.join(logdir, "{}_details".format(label)), "a") as eval_file:
        message = ["Step {}".format(step)]
        for k in score:
            if isinstance(score[k], (int, float)):
                message.append("  {}: {}".format(k, score[k]))
            elif isinstance(score[k], dict):
                message.append("  {}: {}".format(k, ", ".join("{}:{}".format(i, score[k][i]) for i in ["f", "p", "r", "g", "s", "c"])))
        print("\n".join(message), file=eval_file)

def mrp_eval(gold_path, system_path, label, rrhc, mces, parallelize, logdir, logstep, distributed):
    if distributed:
        import os
        os.system("qsub -q cpu*.q -l avx=1 -pe smp {} -N mrp -o /dev/null -j y ".format(parallelize) +
                  "../generated/venv-cpu/bin/python {} ".format(__file__) +
                  "{} {} --label={} --limit_rrhc={} --limit_mces={} --parallelize={} --logdir={} --logstep={} ".format(
                      gold_path, system_path, label, rrhc, mces, parallelize, logdir, logstep) +
                  "'|&' tee -a {}/mtool_evaluate.log".format(logdir))
    else:
        score = mrp_score(mrp_load(path=gold_path), mrp_load(path=system_path), rrhc, mces, parallelize)
        if logdir:
            mrp_tf_log(score, logdir, logstep, label)
        else:
            print(label + " {" + ",\n ".join("\"{}\": {}".format(k, v) for k, v in score.items()) + "}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gold", type=str, help="Gold mrp path")
    parser.add_argument("system", type=str, help="System mrp path")
    parser.add_argument("--distributed", action="store_true", help="Run distributedly")
    parser.add_argument("--label", default="eval", type=str, help="Evaluation label")
    parser.add_argument("--limit_mces", type=int, default=50000, help="Limit for MCES")
    parser.add_argument("--limit_rrhc", type=int, default=2, help="Limit for RRHC")
    parser.add_argument("--logdir", type=str, default=None, help="Log to the given directory")
    parser.add_argument("--logstep", type=int, default=None, help="Log using given step")
    parser.add_argument("--parallelize", type=int, default=1, help="Parallelize")
    args = parser.parse_args()

    mrp_eval(args.gold, args.system, args.label, args.limit_rrhc, args.limit_mces,
             args.parallelize, args.logdir, args.logstep, args.distributed)
