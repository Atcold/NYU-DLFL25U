# NYU introduction to Deep Learning, Fall 2025, Undergrad edition (NYU-DLFL25U)

This repository hosts my new course, introduction to Deep Learning research for undergraduates.
More info can be found on the [companion web site](https://atcold.github.io/NYU-DLFL25U/).

This repo will host:

 - said website (with a table of contents and suggested readings),
 - two notebooks written live in class,
 - the code of selected final projects created by the students, and
 - the project description and key findings (as part of the website).


## Micrograd lectures

Lecture 16 and 17, and corresponding notebooks, introduced a simplified variant of [karpathy/micrograd](https://github.com/karpathy/micrograd), graphically improved for educational purpose.
On Mac, install the graph visualisation dependency with:

```bash
brew install graphviz
pip install graphviz
```

As self-study exercise, you may want to read and understand all the source code found at [karpathy/micrograd/micrograd](https://github.com/karpathy/micrograd/tree/master/micrograd).


## Selected final projects

Follows a list of folder names, project titles and descriptions for some of the students' final projects I deemed interesting.
I suspect most of these projects have been put together with help of a large language model, yet, if the research question and findings are sufficiently solid, I think it's worth checking them out.

The project order reflects the student's total score in the course.

 - [`template`](projects/template/notebook.ipynb): _Project template directory_
    - A sample project structure demonstrating how students can add their contribution via PR.
 - [`backprop_through_depth`](projects/backprop_through_depth/notebook.ipynb): _Backpropagation Through Depth_
    - When optimizing RNNs how can you get rid of exploding/vanishing gradient problem? What if you group gradients by how deep they are? This is what our project explores.