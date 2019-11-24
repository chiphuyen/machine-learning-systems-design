# Machine Learning Systems Design

This booklet covers four main steps of designing a machine learning system:

1. Project setup
2. Data pipeline
3. Modeling: selecting, training, and debugging
4. Serving: testing, deploying, and maintaining

It comes with links to practical resources that explain each aspect in more details. It also suggests case studies written by machine learning engineers at major tech companies who have deployed machine learning systems to solve real-world problems.

At the end, the booklet contains 27 open-ended machine learning systems design questions that might come up in machine learning interviews. The answers for these questions will be published in the book **Machine Learning Interviews**. You can look at and contribute to community answers to these questions on GitHub [here](https://github.com/chiphuyen/machine-learning-systems-design/tree/master/answers). You can read more about the book and sign up for the book's mailing list [here](https://huyenchip.com/2019/07/21/machine-learning-interviews.html).

## Read
To read the booklet, you can clone the repository and find the [HTML](https://github.com/chiphuyen/machine-learning-systems-design/tree/master/build/build1/consolidated.html) and [PDF](https://github.com/chiphuyen/machine-learning-systems-design/tree/master/build/build1/consolidated.pdf) versions in the folder `build`.

## Contribute
This is work-in-progress so any type of contribution is very much appreciated. Here are a few ways you can contribute:

1. Improve the text by fixing any lexical, grammatical, or technical error
1. Add more relevant resources to each aspect of the machine learning project flow
1. Add/edit questions
1. Add/edit answers
1. Other

This book was created using the wonderful [`magicbook`](https://github.com/magicbookproject/magicbook) package. For detailed instructions on how to use the package, see their GitHub repo. The package requires that you have `node`. If you're on Mac, you can install `node` using:

```
brew install node
```

Install `magicbook` with:

```
npm install magicbook
```

Clone this repository:

```
git clone https://github.com/chiphuyen/machine-learning-systems-design.git
cd machine-learning-systems-design
```

After you've made changes to the content, you can build the booklet by the following steps:

```
magicbook build
```

You'll find the HTML and PDF files in the folder `build`.

## Acknowledgment

I'd like to thank Ben Krause for being a great friend and helping me with this draft!


## Citation
