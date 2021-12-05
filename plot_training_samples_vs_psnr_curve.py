import matplotlib.pyplot as plt
import os
import json

if __name__ == '__main__':
    bo_root = os.path.join('runs_fine_tuning/lego-ft-large-bo/version_0')
    random_root = os.path.join('runs_fine_tuning/lego-ft-large-random/version_0')
    res = []
    for root in [bo_root, random_root]:
        val_psnrs = []
        new_view_indices = []
        for file in sorted(os.listdir(root)):
            print(file)
            if 'psnrs_val_' in file:
                with open(os.path.join(root, file), 'r') as f:
                    val_psnr = json.load(f)
                    values = [float(n) for n in val_psnr.values()]
                    val_psnrs.append(sum(values)/len(values))

        with open(os.path.join(root, 'next_view_indices.txt'), 'r') as f:
            next_view_counts = [len(line.strip().split(',')) + 16 for line in f.readlines()]    # + original 16 training pose
        next_view_counts.insert(0, 16)
        res.append([next_view_counts, val_psnrs])

    total_count = min(len(res[0][1]), len(res[1][1]))
    plt.plot(res[0][0][:total_count], res[0][1][:total_count], label="bo")
    plt.plot(res[1][0][:total_count], res[1][1][:total_count], label="random")
    plt.ylabel('val_PSNR')
    plt.xlabel('num_samples')
    plt.legend()
    plt.show()
