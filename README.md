[![Build Status](https://travis-ci.org/sdleffler/collapse.svg?branch=master)](https://travis-ci.org/sdleffler/collapse)

# collapse

The wavefunction collapse algorithm, reimplemented in Rust, with a nice CLI.

Inspired by the reference implementation [here.](https://github.com/mxgmn/WaveFunctionCollapse)

# Features

- *An accurate progress bar!* (the time estimate seems to be based on a first derivative approximation, and this algorithm tends to accelerate over time. So TODO: a different progress bar.)
- More options than the original! Full control over augmenting the sample with symmetries, and over which axes are periodic in the input/output.
- Coming soon: Built-in support for animations and voxels, as well as approximate comparisons instead of exact equality tests when colliding samples. The goal with the approximate equality testing is to make the algorithm work with images that have larger palettes.

# Running/building it

[Cargo](https://crates.io/) is required to compile and run this repository.

To install it, simply run `cargo build --release && cargo install`. The installed binary will be named `runcollapse`. Check out the command-line interface with `runcollapse --help`. Building with `--release` is heavily recommended, as it speeds up runtimes by a ridiculous amount anecdotally determined to be at least ten-fold.

Features:

## `runcollapse 2d`

The two-dimensional case of the wavefunction collapse algorithm. Options:

```
runcollapse-2d
The 2D case of the wavefunction collapse algorithm. Samples an input image and produces an output image.

USAGE:
    runcollapse 2d [FLAGS] [OPTIONS] <INPUT> <OUTPUT> -d <x, y> [--] [ARGS]

FLAGS:
        --all-symmetry    Augment the sample image with rotations/reflections, using all members of the relevant symmetry group. This is the default symmetry setting.
    -h, --help            Prints help information
        --no-symmetry     Do not augment the sample image with rotations/reflections.
    -V, --version         Prints version information

OPTIONS:
    -n <x, y>
            The sample dimensions; expects two positive nonzero integers, corresponding to the width and height of the rectangle to be used for sampling the input. Defaults to 3
            for both axes.
    -d <x, y>
            The dimensions of the output image.
    -p, --periodic-input <x, y>
            The input periodicity; expects two booleans, corresponding to the x and y axes. If true, then the input will be processed as wrapping on that axis. Defaults to false
            for both axes.
    -P, --periodic-output <x, y>
            The output periodicity; expects two booleans, corresponding to the x and y axes. If true, then the output will be processed as wrapping on that axis. Defaults to true
            for both axes.
    -c <source x, source y, output x (optionally a range x0..x1, x0 inclusive, x1 exclusive), output y (optionally a range y0..y1, y0 inclusive, y1 exclusive)>...        

ARGS:
    <INPUT>              The input file to sample.
    <OUTPUT>             The output file to generate.
    <identity>           The original image, since the identity transformation is a no-op. Don't forget this if you're building up a custom set of symmetries.
    <reflect-x>          Reflect over the x axis.
    <reflect-y>          Reflect over the y axis.
    <reflect-y-rot90>    Reflect over the y axis, and then rotate by 90 degrees. This is equivalent to a reflection over the line y = -x.
    <reflect-x-rot90>    Reflect the over the x axis, and then rotate by 90 degrees. This is equivalent to a reflection over the line y = x.
    <rot90>              Rotate the image 90 degrees clockwise.
    <rot180>             Rotate the image 180 degrees clockwise.
    <rot270>             Rotate the image 270 degrees clockwise.
```

To change symmetries, one can either use the flag `--all-symmetries`, `--no-symmetries` to disable all symmetry augmentations other than the original image, or a combination of the `identity`, `reflect-x`, `reflect-y`, etc. options to enable specific symmetries.

An example command, using this specific symmetry selection:

```runcollapse 2d resources/City.png output/City.png -d=64,64 --periodic-input=true,false --periodic-output=true,false identity reflect-x -c 0,-1,0,-1 0,0,0..63,0```

This runs with the input and output wrapping on the x axis only, the identity and horizontally reflected symmetries enabled, and requiring that the pixel at (0, -1) in the input appears at (0, -1) in the output. This translates to the pixel one pixel from the bottom in the input appearing one pixel from the bottom in the output (negative indices wrap, Python-style.) The second constraint requires that the pixel at (0, 0) in the input appears in the *range* from (0, 0) to (63, 0) in the output - so, the entire top row of pixels. The output size is set to be 64 by 64 pixels.

## `runcollapse 2d-anim`

Coming soon - uses the three-dimensional case, and stacks input animation frames to form a 3D volume before running the algorithm on it.

## `runcollapse 3d`

Coming soon - voxels!
