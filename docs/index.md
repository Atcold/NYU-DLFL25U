---
title: Introduction to Deep Learning Research
author: Alfredo Canziani
lang-ref: home
---

<style>
.lesson-box {
    border: 1px solid currentColor;
    border-radius: 8px;
    padding: 0.3em 0.6em;
    margin-top: -1em; /* eat the h2's default bottom margin so the box hugs "Lesson NN" */
}
.lesson-title {
    font-size: 1.5em; /* match the browser-default h2 size used by the theme */
    font-weight: 700;
    line-height: 1.25;
}
.lesson-sub {
    margin-top: 0.2em;
    opacity: 0.55;
    font-style: italic;
}
@media (max-width: 700px) { /* reclaim horizontal space on phones */
    /* shrink the variable, not .page's padding directly, so the menu bar's
       negative margin (calc(0px - var(--page-padding))) stays in sync and
       doesn't overflow the viewport */
    :root { --page-padding: 0px; }
    .content { padding-left: 8px; padding-right: 8px; }
}
</style>

**CSCI-UA 480 075 · FALL 2025 · [NYU COURANT INSTITUTE OF MATHEMATICAL SCIENCES](https://cims.nyu.edu/)**

| INSTRUCTOR  | Alfredo Canziani      |
| LECTURES    | Tue/Thu 12:00 – 13:45 |
| CODE        | [2025 repo](https://github.com/Atcold/NYU-DLFL25U) |
| BLACKBOARDS | [Google Drive](https://drive.google.com/drive/folders/1OLMD0FWh_oKsyXKhWZ5ZfcjgzaAboKNh) |
| READINGS    | [Google Drive](https://drive.google.com/drive/folders/1WAovR_b-rGFDxYy8bN3NssP6MZZXkVET) |
| SLIDES      | [Google Drive](https://drive.google.com/drive/folders/17y84oQafS7P6W4Ja4cJ94Cx0XBa3YNDS) |

This second offering of my new course is meant to be an introduction to Deep Learning research for undergraduate (or advanced high school) students.

The aim of this course is to get the students fluent in reasoning, using:

 - maths (linear algebra, calculus, logic),
 - diagrams and schematics (abstract graphical language),
 - graphs (function plotting and asymptotic behaviour),
 - physics (reducing systems to their base parts to identify emerging collective behaviours), and
 - coding (empirical verification of proposed hypothesis).

 To test the students' knowledge, the course uses 6 quizzes throughout the semester, 4 homework assignments, 2 projects, and a final oral exam, where students are examined on final project significance and originality, project presentation and defence, course content knowledge, and communication effectiveness.

 Selected final projects and code written in class can be found in the GitHub repo; slides, blackboards, and suggested readings can be found on Google Drive.
 All links are provided at the top of this web page.


# Lectures

Legend: 🖥 slides, 📝 notes, 📓 Jupyter notebook, 🎥 YouTube video.

## Lesson 01 [🎥](https://youtu.be/rg4QyMFONNQ)

<div class="lesson-box"><div class="lesson-title">Course intro + McCulloch & Pitts binary neuron</div><div class="lesson-sub">Using maths & coding as languages of research 📐💻</div></div>

**Suggested readings**
- Whitehead & Russell (1910) [*Principia mathematica*](https://archive.org/details/alfred-north-whitehead-bertrand-russel-principia-mathematica.-1)
- McCulloch & Pitts (1943) [*A logical calculus of the ideas immanent in nervous activity*](https://drive.google.com/file/d/13X3wST4-qtkCB1uJW4oAlGUIIChUsXM5/)
- [*Iverson bracket*](https://en.wikipedia.org/wiki/Iverson_bracket)

**Suggested videos**
- Choi (2011) [*Sound of neurons*](https://youtu.be/8bxpz-YEuao)
- Mahdid (2025) [*Exploring "Logical Calculus of Nervous Activity" by McCulloch & Pitts*](https://www.youtube.com/live/BtTs0iwdB8A)

[![Lesson 01 blackboard](https://lh3.googleusercontent.com/d/1TdkMaLSrP0lR8Tq_lmL8jSzEtOxaFH0p)](https://drive.google.com/file/d/1TdkMaLSrP0lR8Tq_lmL8jSzEtOxaFH0p/)

## Lesson 02 [🎥](https://youtu.be/3_e0HVV3nMM)

<div class="lesson-box"><div class="lesson-title">Programming a neural network</div><div class="lesson-sub">Behaviour by design using weights computed with maths 📐🧠</div></div>

**Suggested readings**
- Summerfield (2025) [*These strange new minds*](https://www.penguinrandomhouse.com/books/750406/these-strange-new-minds-by-christopher-summerfield/)

[![Lesson 02 blackboard](https://lh3.googleusercontent.com/d/1QekWhROt7Yz-JayTk1nxXEaZtJNJwdjU)](https://drive.google.com/file/d/1QekWhROt7Yz-JayTk1nxXEaZtJNJwdjU/)

## Lesson 03 [🎥](https://youtu.be/8WDOAXaxwlU)

<div class="lesson-box"><div class="lesson-title">Wiener's cybernetics, Hebbian plasticity, and Rosenblatt's perceptron</div><div class="lesson-sub">When physical machines start learning 🔁</div></div>

**Suggested readings**
- Wiener (1948) [*Cybernetics*](https://dn790006.ca.archive.org/0/items/norbert-wiener-cybernetics/Norbert_Wiener_Cybernetics_text.pdf)
- Whitehead & Russell (1910) [*Principia mathematica*](https://archive.org/details/alfred-north-whitehead-bertrand-russel-principia-mathematica.-1)
- Gertner (2012) [*The idea factory*](https://www.penguinrandomhouse.com/books/303275/the-idea-factory-by-jon-gertner/)
- Mauchly & Eckert (1945) [*ENIAC*](https://en.wikipedia.org/wiki/ENIAC)
- [*Computer terminal*](https://en.wikipedia.org/wiki/Computer_terminal)
- Intel (1974) [*Intel 8080*](https://en.wikipedia.org/wiki/Intel_8080)
- Monty Python (1969) [*Monty Python's Flying Circus*](https://en.wikipedia.org/wiki/Monty_Python%27s_Flying_Circus)
- Monty Python (1970) [*Spam* (Monty Python sketch)](https://en.wikipedia.org/wiki/Spam_(Monty_Python_sketch))
- van Rossum (1991) [*Python*](https://en.wikipedia.org/wiki/Python_(programming_language))

[![Lesson 03 blackboard 1](https://lh3.googleusercontent.com/d/1J4U0IwMkuRLQ-SHDepmnRlGBFk_RcK6b)](https://drive.google.com/file/d/1J4U0IwMkuRLQ-SHDepmnRlGBFk_RcK6b/)
[![Lesson 03 blackboard 2](https://lh3.googleusercontent.com/d/1uov982b5bXUOp2yf_H6GWb1Kk-ctUa0Q)](https://drive.google.com/file/d/1uov982b5bXUOp2yf_H6GWb1Kk-ctUa0Q/)

## Lesson 04 [🎥](https://youtu.be/DtP2HYp9cNM)

<div class="lesson-box"><div class="lesson-title">Bias, perceptron properties, and multi-class classification</div><div class="lesson-sub">Bias shifts the boundary; more neurons slice the world 📐🧠</div></div>

[![Lesson 04 blackboard 1](https://lh3.googleusercontent.com/d/1Q541DC-nQCgHfEd5goRSmG_rOItG8bEH)](https://drive.google.com/file/d/1Q541DC-nQCgHfEd5goRSmG_rOItG8bEH/)
[![Lesson 04 blackboard 2](https://lh3.googleusercontent.com/d/1lMB-GEic5D4bATAWznofC0482NxoGZOo)](https://drive.google.com/file/d/1lMB-GEic5D4bATAWznofC0482NxoGZOo/)

## Lesson 05 [🎥](https://youtu.be/DYtEA4FTCgE)

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part I: probabilities</div><div class="lesson-sub">Replacing certainty 🌗 with a degree of belonging 📊</div></div>

**Suggested readings**
- [*Iverson bracket*](https://en.wikipedia.org/wiki/Iverson_bracket)
- [*Temperature*](https://en.wikipedia.org/wiki/Temperature)
- [*Thermodynamic β*](https://en.wikipedia.org/wiki/Thermodynamic_beta)

[![Lesson 05 blackboard 1](https://lh3.googleusercontent.com/d/1fJzySpr1PnR8vRJOhtyUJGaaBEGDpXwN)](https://drive.google.com/file/d/1fJzySpr1PnR8vRJOhtyUJGaaBEGDpXwN/)
[![Lesson 05 blackboard 2](https://lh3.googleusercontent.com/d/1bEQYE4GCo2DznHWqzdLNzBovkYW7No6r)](https://drive.google.com/file/d/1bEQYE4GCo2DznHWqzdLNzBovkYW7No6r/)

## Lesson 06 [🎥](https://youtu.be/6urnjbulYt0)

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part II: likelihood and loss</div><div class="lesson-sub">Cross-entropy turns belief into a training signal 📐📊</div></div>

**Suggested readings**
- Watt, Borhani & Katsaggelos (2020) [*Machine learning refined* (2nd ed), § 6.2 *Logistic regression and the cross-entropy cost*](https://www.mlrefined.com/)

[![Lesson 06 blackboard 1](https://lh3.googleusercontent.com/d/1z_0_zvYhYWlkx9U0bQmBcl8PEopgMJo5)](https://drive.google.com/file/d/1z_0_zvYhYWlkx9U0bQmBcl8PEopgMJo5/)
[![Lesson 06 blackboard 2](https://lh3.googleusercontent.com/d/1y8wWpJtx2J5nboVpue8PCw4t-OKNBIJ3)](https://drive.google.com/file/d/1y8wWpJtx2J5nboVpue8PCw4t-OKNBIJ3/)

## Lesson 07 [🎥](https://youtu.be/2PlFRMWDQmQ)

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part III: gradient descent</div><div class="lesson-sub">One ∇ vector, two answers: where to go 🧭 and how fast 🏃💨</div></div>

**Suggested readings**
- [*Gradient descent*](https://en.wikipedia.org/wiki/Gradient_descent) (animation)
- Stewart (2020) [*Calculus: early transcendentals* (9th ed), § 14.1 & § 14.6](https://www.stewartcalculus.com/_update/20/home.html)

[![Lesson 07 blackboard 1](https://lh3.googleusercontent.com/d/1R2GCdgUSV42mbrtgl4KYqAoaDEHxRTah)](https://drive.google.com/file/d/1R2GCdgUSV42mbrtgl4KYqAoaDEHxRTah/)
[![Lesson 07 blackboard 2](https://lh3.googleusercontent.com/d/1c7jzz5Bkas7ZwLPLs4pXWP_u52aWe6Vz)](https://drive.google.com/file/d/1c7jzz5Bkas7ZwLPLs4pXWP_u52aWe6Vz/)

## Lesson 08

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part IV: hardening and multi-class</div></div>

## Lesson 09

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part V: multi-class likelihood and loss</div></div>

## Lesson 10

<div class="lesson-box"><div class="lesson-title">A softer perceptron, part VI: soft-stuff and multi-class SGD</div></div>

## Lesson 11

<div class="lesson-box"><div class="lesson-title">Loss zoo and the least-squares solution</div></div>

## Lesson 12

<div class="lesson-box"><div class="lesson-title">Adaline, first NN winter, and adaptive filters for system identification</div></div>

## Lesson 13

<div class="lesson-box"><div class="lesson-title">Inverse modelling with adaptive filters, 1980s historical background</div></div>

## Lesson 14

<div class="lesson-box"><div class="lesson-title">Learning the feature vector with back-propagation</div></div>

## Lesson 15

<div class="lesson-box"><div class="lesson-title">N per-sample losses and the back-propagation algorithm</div></div>

## Lesson 16

<div class="lesson-box"><div class="lesson-title">Backprop example, on the blackboard and with a Python class</div></div>

## Lesson 17

<div class="lesson-box"><div class="lesson-title">Gradient accumulation</div></div>

## Lesson 18

<div class="lesson-box"><div class="lesson-title">Learning the feature vector, part V: spiral 'despiralisation'</div></div>

## Lesson 19

<div class="lesson-box"><div class="lesson-title">Nonlinear classification with neural nets</div></div>

## Lesson 20

<div class="lesson-box"><div class="lesson-title">Supervised learning with PyTorch</div></div>

## Lesson 21

<div class="lesson-box"><div class="lesson-title">Natural signals and convolutional neural networks</div></div>

## Lesson 22

<div class="lesson-box"><div class="lesson-title">ConvNets for 2D signals, history, and recurrent nets</div></div>

## Lesson 23

<div class="lesson-box"><div class="lesson-title">Project 1: digit captioning</div></div>

## Lesson 24

<div class="lesson-box"><div class="lesson-title">Statistical Natural Language Processing (NLP)</div></div>

## Lesson 25

<div class="lesson-box"><div class="lesson-title">Neural NLP</div></div>

## Lesson 26

<div class="lesson-box"><div class="lesson-title">Attention-based NLP</div></div>
