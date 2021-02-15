import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def visualize():
    # Load
    results = pd.concat( [pd.read_pickle('results.pandas')] )
    pipelines = list(pd.unique(results.pipeline))

    # Print
    print(results)

    ## Bar chart
    fig = plt.figure(figsize=[16, 8])
    sns.stripplot(data=results, y='score', x='pipeline', jitter=True,
                  alpha=.5, zorder=1, palette="Set1")
    sns.pointplot(data=results, y='score', x='pipeline',
                  zorder=1, palette="Set1")
    plt.ylabel('Score')
    plt.ylim([0.2, 0.8])
    plt.grid()
    plt.tight_layout()
    fig.savefig("fig-bar.png", dpi=150)

    ## Paired plots
    for i in range(len(pipelines)):
        for j in range(i+1, len(pipelines)):
            fig = plt.figure(figsize=[4, 4])
            paired = results.pivot_table(values='score', columns='pipeline',
                                         index=['subject', 'session'])
            paired = paired.reset_index()

            sns.regplot(data=paired, y=pipelines[i], x=pipelines[j],
                        fit_reg=False)
            plt.plot([0, 1], [0, 1], ls='--', c='k')
            plt.xlim([0.3, 1])
            plt.tight_layout()
            fig.savefig("fig-reg-%d-vs-%d.png" % (i, j), dpi=150)


if __name__ == "__main__":
    visualize()
