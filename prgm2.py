import matplotlib.pyplot as plt
x = [1, 2, 3]
y = [2, 4, 1]
plt.plot(x, y)
plt.xlabel('x - axis')
plt.ylabel('y - axis')
plt.title('My first graph!')
plt.show()
import plotly.express as px
fig = px.bar(x=["a", "b", "c"], y=[1, 3, 2])
fig.show()
from bokeh.plotting import figure, output_file, show
graph = figure(title = "Bokeh Line Graph")
u = [1, 2, 3, 4, 5]
v = [5, 4, 3, 2, 1]
graph.line(u, v)
show(graph)
import seaborn as sns
data = sns.load_dataset("iris")
sns.lineplot(x="sepal_length", y="sepal_width", data=data)
plt.title('Title using Matplotlib Function')
plt.show()
