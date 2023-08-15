"""
╦  ┬┬  ┬┌─┐  ╔╦╗┌─┐┌─┐┌┬┐
║  │└┐┌┘├┤    ║ ├┤ └─┐ │
╩═╝┴ └┘ └─┘   ╩ └─┘└─┘ ┴
    First inspiration code from here:
    https://makersportal.com/blog/2018/8/14/real-time-graphing-in-python
"""
import matplotlib.pyplot as plt
import numpy as np
from db import MyDatabase
import pandas as pd

# use ggplot style for more sophisticated visuals
plt.style.use("ggplot")


def live_plotter(x_vec, y1_data, line1, identifier="", pause_time=0.1):
    if line1 == []:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13, 6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        (line1,) = ax.plot(x_vec, y1_data, "-o", alpha=0.8)
        # update plot label/title
        plt.ylabel("Y Label")
        plt.title("Title: {}".format(identifier))
        plt.show()

    plt.xlim(min(x_vec), max(x_vec))
    # after the figure, axis, and line are created, we only need to update the y-data
    # line1.set_ydata(y1_data)
    line1.set_data(x_vec, y1_data)

    # adjust limits if new data goes beyond bounds
    if (
        np.min(y1_data) <= line1.axes.get_ylim()[0]
        or np.max(y1_data) >= line1.axes.get_ylim()[1]
    ):
        plt.ylim([np.min(y1_data) - np.std(y1_data), np.max(y1_data) + np.std(y1_data)])
    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)

    # return line so we can update it again in the next iteration
    return line1


if __name__ == "__main__":
    size = 20
    x_vec = np.linspace(0, 1, size + 1)[0:-1]
    y_vec = np.zeros(len(x_vec))
    line1 = []
    dbms = MyDatabase("sqlite", dbname="mydb.sqlite")
    while True:
        # TODO: This brings in the whole table. It will not scale well... Maybe an sql fetching the last n rows?
        df = pd.read_sql_table(
            "bg_values", dbms.db_engine, parse_dates=["timestamp"], index_col="id"
        )
        df["value"] = pd.to_numeric(df["value"], downcast="float")
        df_plot = df.tail(20)
        # y_vec[-1] = rand_val
        line1 = live_plotter(
            df_plot["timestamp"], df_plot["value"], line1, pause_time=0.5
        )
        # y_vec = np.append(y_vec[1:], 0.0)
