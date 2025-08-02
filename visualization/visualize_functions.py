import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()  # <- Turn off interactive mode

import matplotlib.gridspec as gridspec





def table(channel_handling, df_avg, df_subset):
    list_pred_len = [96, 192, 336, 720, "Avg"]
    list_data_path = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather", "traffic", "electricity"]
    index_labels = [f"{data_path} - {pred_len}" for data_path in list_data_path for pred_len in list_pred_len]

    list_pred_len = [24, 36, 48, 60, "Avg"]
    list_data_path = ["national_illness"]
    index_labels = index_labels + [f"{data_path} - {pred_len}" for data_path in list_data_path for pred_len in list_pred_len]

    # Example: define some columns
    columns = [f"{model} - {metric}" for model in ["Linear_final", "ModernTCN", "PatchTST"] for metric in ["MSE", "MAE"]]

    df_table = pd.DataFrame(index=index_labels, columns=columns)

    for data_path in ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "exchange_rate", "weather", "traffic", "electricity", "national_illness"]:
        if data_path == "national_illness":
            list_pred_len = [24, 36, 48, 60, "Avg"]
        else:
            list_pred_len = [96, 192, 336, 720, "Avg"]

        for pred_len in list_pred_len:
            for model in ["Linear_final", "ModernTCN", "PatchTST"]:
                if pred_len == "Avg":
                    df_filtered = df_avg[
                        (df_avg['data_path'].str.contains(data_path)) &
                        (df_avg['model'] == model) &
                        (df_avg['channel_handling'] == channel_handling)
                    ]
                else:
                    df_filtered = df_subset[
                        (df_subset['data_path'].str.contains(data_path)) &
                        (df_subset['pred_len'] == pred_len) &
                        (df_subset['model'] == model) &
                        (df_subset['channel_handling'] == channel_handling)
                    ]
                    #display(df_filtered)

                if not df_filtered.empty:
                    mse = df_filtered['mse'].values[0]
                    mae = df_filtered['mae'].values[0]
                    df_table.loc[f"{data_path} - {pred_len}", f"{model} - MSE"] = round(mse, 3)
                    df_table.loc[f"{data_path} - {pred_len}", f"{model} - MAE"] = round(mae, 3)

    return df_table

def render_table(df, title="", fontsize=10, show:bool=False):
    fig, ax = plt.subplots(figsize=(len(df.columns)*1.8, len(df)*0.3))  # Adjust figure size
    ax.axis('off')  # Hide axes

    table = plt.table(
        cellText=df.values,
        rowLabels=df.index,
        colLabels=df.columns,
        cellLoc='center',
        loc='center'
    )

    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.2, 1.2)  # Adjust scaling for better readability

    if title:
        plt.title(title, fontsize=fontsize+2)

    save_dir = f"plots/tables"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{title}.png"
    plt.savefig(save_path, bbox_inches="tight")

    plt.tight_layout()
    
    plt.close(fig)
    plt.close('all')


def channel_wise(df_subset, pred_len=24, data_path="national_illness", model="Linear_final", m_type= [["CI_glob", 0], ["CI_loc", 0], ["CD", 0], ["Delta", 0]], show:bool=False):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharey=True)

    if model=="PatchTST":
        rows = [0]
    else:
        rows = [0, 1]
    
    for row in rows:
        for col in [0, 1]:
            ax = axs[row, col]
            channel_handling = m_type[2*row+col][0]
            cd_weight_decay = m_type[2*row+col][1]

            df_CI = df_subset[
                (df_subset['model'] == model) &
                (df_subset['channel_handling'] == channel_handling) &
                (df_subset['cd_weight_decay'] == cd_weight_decay) &
                (df_subset['data_path'].str.contains(data_path)) &
                (df_subset['pred_len'] == pred_len)
            ].copy()

            #display(df_CI)
            
            # Extract lists
            data_train = df_CI["mse_train_per_channel_list"].tolist()
            data_test = df_CI["mse_per_channel_list"].tolist()

            # Combine per channel
            combined = [[train, test] for train, test in zip(data_train[0], data_test[0])]

            data = list(map(list, zip(*combined)))  # shape: 2 × 7

            group_labels = ['train', 'test']
            colors = ['#1f77b4', '#ff7f0e']  # train, test
            bar_width = 0.35
            x = np.arange(len(combined))  # 7 channels

            # Plot on axs[0, 0]
            for i, group in enumerate(data):
                offset = (i - len(data)/2 + 0.5) * bar_width
                ax.bar(x + offset, group, width=bar_width, label=group_labels[i], color=colors[i])

            # Horizontal lines
            ax.axhline(y=df_CI["mse_train"].values[0], color=colors[0], linestyle='--', linewidth=1)
            ax.axhline(y=df_CI["mse"].values[0], color=colors[1], linestyle='--', linewidth=1)

            # Formatting
            ax.set_xticks(x)
            ax.set_xticklabels([f'Ch {i+1}' for i in x])
            #ax.set_xticklabels([f'{i+1}' for i in x])
            #ax.set_ylabel("MSE")
            if channel_handling == "CI_glob" or channel_handling == "CI_loc":
                ax.set_title(channel_handling)
            else:
                ax.set_title(f"{channel_handling} (cd_weight_decay={cd_weight_decay})")
            #ax.set_xlabel("Channels")
            ax.grid(alpha=0.25)
            ax.legend()
            
    fig.suptitle(f"{model} | {data_path} | pred_len={pred_len}", fontsize=10)
    fig.tight_layout()
    
    if pred_len == "Avg":
        save_dir = f"plots/final_channel_wise/{model}/Avg"
    else:
        save_dir = f"plots/final_channel_wise/{model}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{data_path}_{pred_len}.png"
    plt.savefig(save_path)
    
    plt.close(fig)
    plt.close('all')

    return None



def channel_handling(df_subset, data_path="ETTh1", pred_len=96, model="Linear_final", show:bool=False):
    channel_handling_list = ["CI_glob", "CI_loc", "CD", "Delta"]

    fig = plt.figure(figsize=(16, 4))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 4, 4], wspace=0.3)
    axes = [plt.subplot(gs[i]) for i in range(4)]

    bar_width = 0.4
    colors = ['#1f77b4', '#ff7f0e']  # blue (train), orange (test)

    for col, ch_handling in enumerate(channel_handling_list):
        ax = axes[col]

        df_CI = df_subset[
            (df_subset['model'] == model) &
            (df_subset['channel_handling'] == ch_handling) &
            (df_subset['data_path'].str.contains(data_path)) &
            (df_subset['pred_len'] == pred_len)
        ].copy()

        # Extract values
        cd_labels = df_CI['cd_weight_decay'].astype(str).tolist()
        data_train = df_CI["mse_train"].tolist()
        data_test = df_CI["mse"].tolist()

        x = np.arange(len(cd_labels))

        # Plot grouped bars
        ax.bar(x - bar_width/2, data_train, width=bar_width, color=colors[0], label='Train')
        ax.bar(x + bar_width/2, data_test,  width=bar_width, color=colors[1], label='Test')

        # Labeling
        ax.set_title(ch_handling)
        ax.set_xlabel("CD Reg.")
        ax.set_xticks(x)
        ax.set_xticklabels(cd_labels, rotation=45)
        ax.legend()

        if col == 0:
            ax.set_ylabel("MSE")

    # Collect all values
    all_mse = df_subset[
        (df_subset['model'] == model) &
        (df_subset['data_path'].str.contains(data_path)) &
        (df_subset['pred_len'] == pred_len)
    ][["mse", "mse_train"]].values.flatten()

    y_max = np.max(all_mse)

    # Share y-axis across subplots
    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
        ax.set_ylim(0, y_max*1.05)

    fig.suptitle(f"{model} | {data_path} | pred_len={pred_len}", fontsize=10)
    #plt.savefig("plots/final_channel_handling/"+model+"/"+data_path+"_"+str(pred_len)+".png")

    if pred_len == "Avg":
        save_dir = f"plots/final_channel_handling/{model}/Avg"
    else:
        save_dir = f"plots/final_channel_handling/{model}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{data_path}_{pred_len}.png"
    plt.savefig(save_path)

    plt.close(fig)
    plt.close('all')

    return None



def calc_avg(df_avg, df_subset, data_path="national_illness", model="Linear_final"):
    random_seeds = df_subset['random_seed'].unique().tolist()
    list_channel_handling = df_subset['channel_handling'].unique().tolist()

    for random_seed in random_seeds:
        df_seed = df_subset[
            (df_subset['model'] == model) &
            (df_subset['data_path'].str.contains(data_path)) &
            (df_subset['random_seed'] == random_seed)
        ].copy()

        if df_seed.empty:
            continue
        
        for channel_handling in list_channel_handling:
            df_seed_channel = df_seed[df_seed['channel_handling'] == channel_handling]
            if df_seed_channel.empty:
                continue

            list_cd_weight_decay = df_seed_channel['cd_weight_decay'].unique().tolist()
            
            for cd_weight_decay in list_cd_weight_decay:
                df_seed_channel_cd = df_seed_channel[df_seed_channel['cd_weight_decay'] == cd_weight_decay]
                if df_seed_channel_cd.empty:
                    continue

                # Calculate average values for each channel handling and cd_weight_decay
                avg_row = {
                    'model': model,
                    'data_path': data_path,
                    'pred_len': "Avg",
                    'random_seed': random_seed,
                    'mse': df_seed_channel_cd['mse'].mean(),
                    'mae': df_seed_channel_cd['mae'].mean(),
                    'mse_train': df_seed_channel_cd['mse_train'].mean(),
                    'mae_train': df_seed_channel_cd['mae_train'].mean(),
                    'cd_weight_decay': cd_weight_decay,
                    'channel_handling': channel_handling,
                    'Count': len(df_seed_channel_cd),
                }

                # Add per channel averages
                avg_row['mse_per_channel_list'] = [np.mean(x) for x in zip(*df_seed_channel_cd['mse_per_channel_list'].tolist())]
                avg_row['mae_per_channel_list'] = [np.mean(x) for x in zip(*df_seed_channel_cd['mae_per_channel_list'].tolist())]
                avg_row['mse_train_per_channel_list'] = [np.mean(x) for x in zip(*df_seed_channel_cd['mse_train_per_channel_list'].tolist())]
                avg_row['mae_train_per_channel_list'] = [np.mean(x) for x in zip(*df_seed_channel_cd['mae_train_per_channel_list'].tolist())]

                avg_row['mse_per_channel_std'] = np.std(avg_row['mse_per_channel_list'])
                avg_row['mae_per_channel_std'] = np.std(avg_row['mae_per_channel_list'])
                avg_row['mse_train_per_channel_std'] = np.std(avg_row['mse_train_per_channel_list'])
                avg_row['mae_train_per_channel_std'] = np.std(avg_row['mae_train_per_channel_list'])

                avg_row['mse_per_channel_range'] = max(avg_row['mse_per_channel_list']) - min(avg_row['mse_per_channel_list'])
                avg_row['mae_per_channel_range'] = max(avg_row['mae_per_channel_list']) - min(avg_row['mae_per_channel_list'])
                avg_row['mse_train_per_channel_range'] = max(avg_row['mse_train_per_channel_list']) - min(avg_row['mse_train_per_channel_list'])
                avg_row['mae_train_per_channel_range'] = max(avg_row['mae_train_per_channel_list']) - min(avg_row['mae_train_per_channel_list'])

                df_avg = pd.concat([df_avg, pd.DataFrame([avg_row])], ignore_index=True)

    return df_avg