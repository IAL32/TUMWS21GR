# TUM Winter Semester 2021/2022 - Guided Research

This repository serves as a way to track my progress during the guided research.

The whole ordeal is supervised by [M.Sc. Alexander Isenko](https://www.in.tum.de/i13/team/alexander-isenko/), part of the chair of Business Information Systems at the Technische Universität München.

<!-- / -->

# Project Description

## Background

The project stems from the unpublished (as of today, September 2021) paper of Alexander, which analyzes bottlenecks of the preprocessing step in deep learning pipelines and provides a profiling library that automatically selects a preprocessing strategy according to the input data. When in place, this approach has brought an increased throughput of up to 6.2x compared to untuned systems. As the paper states, space consumption is a big issue, because it impacts general throughput of the pipeline with problems such as network and I/O latency. We would like to find out if we can further increase the performance of the library by applying compression algorithms in-between the preprocessing pipeline steps.

## Goals

The goals of this project is to provide further insights regarding the use of compression algorithms in-between the preprocessing pipeline steps. The basic idea is that compressing the data before a step of the pipeline reduces the strain on the network and storage device as less amount of data is used, thus increasing the overall throughput of the system.

## Abstract

Deep learning models are leading to several challenges to resource optimization, as the amount of data, they require to be trained increases along with their complexity. Consequently, data-intensive steps such as preprocessing are being studied to enhance the performance of training pipelines. One of these studies [1] has reportedly achieved good performance by building an open-source profiling library that can automatically decide suitable preprocessing strategies based on a cost model, and the results it has achieved show the potential of pipeline tuning. Compressing data is a process that is being used for different systems to save storage and network bandwidth. This however leads to additional processing power overhead on the system that receives the compressed data, which has to run a potentially very slow decompression algorithm.

The preprocessing pipeline can be split into steps which are run *once*, called "offline" from this point onwards, and steps which are performed *every iteration*, called "online". The set of preprocessing steps depends on the dataset and the model input, but generally, any transformation is a step, like cropping an image or encoding a word. A preprocessing *strategy* is processing up to (and including) a step offline, and the remainder of the pipeline is executed online.

We choose to explore the type of data that was used to analyze the original study's preprocessing pipelines. In particular, we will start with images (Computer Vision problems). Selecting this approach allows us to avoid unnecessary additional work, as the datasets and parts of the software are already present and working.

Our starting approach is simple: at every step *n*, the output data will be compressed using an algorithm, and at the next step *n+1*, the input data from the step *n* is decompressed. Our intuition is that we reduce the strain on the bandwidth and disk usage, at the cost of processing performance.

In the case a particular algorithm at a particular step of the preprocessing pipeline does not yield particular benefits to the overall performance of the system, then we can consider applying other algorithms. For example, it is very well possible that certain algorithms used at different steps of the pipeline could carry more performance benefits than others. Some can be more efficient when used against structured data.

Example of possible compression algorithms and formats that can be applied are the ones based of `LZ77`, which is one of the most widely used algorithms for data compression, although it is ill-advised to use them as one-size-fits-all solutions. Another very common compression algorithm is `gzip` for dealing with file data. It is often seen as one of the low-hanging fruits to pick when trying to reduce strain on bandwidth in HTTP communications. Furthermore, compression algorithms that target floating-point and integer data, which can have the most impact when dealing with multidimensional data, such as `SIMD-BP128` or `varint-G81U` [2] can be used, and other algorithms that can be applied according to the nature of the input data.

Finally, the guided research presented here aims to provide further insights into the above-mentioned profiling library by exploring the effects of data compression on storage and network usage. We will provide an in-depth analysis of the impact of compression algorithms applied to different data representations of preprocessing pipelines.

[1] Alexander Isenko et al. 2021 Where Is My Training Bottleneck?
Hidden Trade-Offs in Deep Learning Preprocessing Pipelines

[2] Daniel Lemire and Leonid Boytsov. 2015. Decoding billions of integers per second through vectorization. Software: Practice and Experience 45, 1 (2015), 1–29.

# 08/10/2021

## First steps

In the past few days, I have started setting up the virtual machine to work on for the guided research. I have been given an account to use LRZ resources at the following link: https://openstack.msrg.in.tum.de/horizon/auth/login/.

In order to access the VM, I had to install Cisco AnyConnect VPN (which you can [setup using this guide](https://www.lrz.de/services/netz/mobil/vpn_en/anyconnect_en/)), and use `asa-cluster.lrz.de` as the address. To login to the VPN I have used my TUM ID together with my TUM ID password. The TUM ID had to be preceded by `!` (exclamation mark).

After that, you should be able to access the Openstack instance, as well as the VM you will create.

First, generate an SSH key pair, that you will use to connect to the VM (on Linux you can use `ssh-keygen`). Then, copy and paste the public key (probably under `~/.ssh/id_rsa.pub`) in the page under the menu **Compute → Access & Security → Key Pairs**.

Due to some unknown problems with the VM creation, it is suggested to spin up 4 VMs, check the logs and see which one does not have the error `ci-info: no authorized SSH keys fingerprints found for user ubuntu.`. Select one if there are multiple without errors, and delete the other ones.

Note down the IP, which should be something like `172.xxx.xxx.xxx`, and connect to it using your SSH key and the user `ubuntu` with the following command:

```
ssh -i ~/.ssh/id_rsa ubuntu@172.xxx.xxx.xxx
```

If the connection works and you log in inside the VM, woohoo! Works. Otherwise, check with someone.

To ease your connection to the server, change your config file in `~/.ssh/config` and add the following entry:

```config
Hostname custom-virtual-machine-name
    User ubuntu
    IdentityFile ~/.ssh/id_rsa
```

## Getting some work done

The first steps in the VM are rather simple. I got access to the `presto` library source code in https://gitlab.i13.in.tum.de/alexander_isenko/presto, and `pbr-prototype` in https://gitlab.i13.in.tum.de/alexander_isenko/pbr-prototype.

I then used `git clone` to clone the repositories inside the VM, and followed the README instructions to set up the *miniconda* virtual environment.

## Baby steps

The first step was to write the abstract, which is inside this very document at the beginning. Next, I need to explore the following topics:
- get to know `zlib` and `gzip` compression libraries
- find the C++ implementation in Tensorflow that implements such compression libraries, find if we can change these tuning parameters and try to change them

# 10/10/2021

## A quick look at GZIP

[GZIP](https://www.gnu.org/software/gzip/manual/gzip.html) is a compression software based on LZ77. The [documentation](https://www.gnu.org/software/gzip/manual/gzip.html) shows tons of stuff, but here we are just going to focus on parameters that affect compression rate and speed.

> `gzip` uses the Lempel–Ziv algorithm used in zip and PKZIP. The amount of compression obtained depends on the size of the input and the distribution of common substrings. **Typically, text such as source code or English is reduced by 60–70%.** Compression is generally much better than that achieved by LZW (as used in compress), Huffman coding (as used in pack), or adaptive Huffman coding (compact).  
Compression is always performed, even if the compressed file is slightly larger than the original. The worst case expansion is a few bytes for the gzip file header, plus 5 bytes every 32K block, or an expansion ratio of 0.015% for large files.

This is the help output:

```
Usage: gzip [OPTION]... [FILE]...
Compress or uncompress FILEs (by default, compress FILES in-place).

Mandatory arguments to long options are mandatory for short options too.

  -c, --stdout      write on standard output, keep original files unchanged
  -d, --decompress  decompress
  -f, --force       force overwrite of output file and compress links
  -h, --help        give this help
  -k, --keep        keep (don't delete) input files
  -l, --list        list compressed file contents
  -L, --license     display software license
  -n, --no-name     do not save or restore the original name and timestamp
  -N, --name        save or restore the original name and timestamp
  -q, --quiet       suppress all warnings
  -r, --recursive   operate recursively on directories
      --rsyncable   make rsync-friendly archive
  -S, --suffix=SUF  use suffix SUF on compressed files
      --synchronous synchronous output (safer if system crashes, but slower)
  -t, --test        test compressed file integrity
  -v, --verbose     verbose mode
  -V, --version     display version number
  -1, --fast        compress faster
  -9, --best        compress better

With no FILE, or when FILE is -, read standard input.

Report bugs to <bug-gzip@gnu.org>.
```

What's interesting for us in this case are the arguments `--fast` and `--best`, and more generally `-n`, where `n` regulates the speed of compression in a range from `1` (faster compression, and lowest compression ratio) to `9` (slower compression, and highest compression ratio).

Different benchmark sources [[1](https://www.rootusers.com/gzip-vs-bzip2-vs-xz-performance-comparison/), [2](https://bbengfort.github.io/2017/06/compression-benchmarks/)] show different aspects about gzip, which could be of use in the future. For example, it is worth noting that this algorithm is always amongst the most "stable" ones, meaning that compression and extraction times are not so much affected by the size of the input file, whilst keeping the compression ratio linear.

Thus, I figure that this algorithm can be used as a reference benchmark for other types of compression algorithms that we are going to try in the future.

## A quick look at ZLib.net

> zlib is designed to be a free, general-purpose, legally unencumbered -- that is, not covered by any patents -- lossless data-compression library for use on virtually any computer hardware and operating system. The zlib data format is itself portable across platforms. Unlike the LZW compression method used in Unix compress(1) and in the GIF image format, the compression method currently used in zlib essentially never expands the data. (LZW can double or triple the file size in extreme cases.) zlib's memory footprint is also independent of the input data and can be reduced, if necessary, at some cost in compression.

I am taking the introduction from the [zlib website](https://zlib.net). Difference from GZip and ZLib can be found in a [StackOverflow post from Mark Adler himself](https://stackoverflow.com/a/20765054/12482799).

# 12/10/2021

## Looking at Tensorflow

After getting the basics of `gzip` and `zlib`, we can start and see where Tensorflow actually enables some compression.

`presto`'s code uses `TFRecordDataset` for handling datasets, with options for compression and decompression. After some looking, I can make a small map of what connects to what, so that we can perhaps try our own implementation of a compression type.

`TFRecordDataset` (`tensorflow/core/kernels/data/tf_record_dataset_op.cc`) is the class that holds the dataset. It can also be iterated through.

`TFRecordWriter` (`tensorflow/core/data/snapshot_utils.cc:119`) is the class that writes `TFRecordDataset` instances as `.tfrecord` files. It accepts `const std::string& filename, const std::string& compression_type` as arguments, and writes using `TFRecordWriter::WriteTensors()`.

`RecordWriter` (`tensorflow/core/lib/io/record_writer.cc`) is the generic class that writes the files, and this is what we are interested in. According to the compression type (`tensorflow::io::compression::kZlib`, `compression::kGzip`, `compression::kNone`, `compression::kSnappy`), it will select the proper writer.

`ZlibOutputBuffer` (`tensorflow/core/lib/io/zlib_outputbuffer.cc`) is the specific implementation of `RecordWriter` and wrapper around the `zlib` library. It uses the `deflate` method and returns a buffer that `RecordWriter` can then use. Note that using `GZIP` as the compression type will still use `ZlibOutputBuffer`, but with a slight change in the `window_bits` option, as noted in `tensorflow/core/lib/io/zlib_compression_options.h:132`.

Similarly, we can track down `TFRecordReader` all the way down to `ZlibInputStream`.

# 21/10/2021

## Building Tensorflow from source

So, the first thing that we have to do in order for us to actually start modifying the code is to get Tensorflow building correctly into our remote node. We are taking as a reference the [official tensorflow documentation](https://www.tensorflow.org/install/source).

From a previous run, we can use a handy script that installs `miniconda` into the local machine. Look at the [presto library](https://gitlab.i13.in.tum.de/alexander_isenko/presto) for the script.

Once inside miniconda, before creating a virtual environment, we have to download the repo and checkout to a release version (this will make things much easier down the road), such as `r2.6` (11 August 2021):

```
git clone git@github.com:tensorflow/tensorflow.git
git checkout r2.6
```

Now, let's create the virtual environment:

```
conda create --name tensorflow
```

And then enter the environment:

```
conda activate tensorflow
```

And navigate into the folder:

```
cd tensorflow/
```

### Installing dependencies

First thing once inside the venv (virtual environment), is to install dependencies:

```
pip install -U pip numpy wheel
pip install -U keras_preprocessing --no-deps
```

Then install Bazel. To install it, we are going to install `Bazelisk` first, which is kind of similar to `nvm` (Node version manager), but for `Bazel`.

We will need to install `npm` to have a working `Bazelisk` installation, so:

```
sudo apt install npm -y
```

And then we can install bazelisk (installing it automatically installs `bazel` as well):

```
sudo npm install -g @bazel/bazelisk
```

The executable will already be available to the `PATH` by using the `-g` (`--global`) flag.

### Configure

Now run the configure script:

```
python configure.py
```

It will ask a lot of stuff. For now, we will just press `Enter` to everything (No to everything). The final output will be something like:

```
Preconfigured Bazel build configs. You can use any of the below by adding "--config=<>" to your build command. See .bazelrc for more details.
        --config=mkl            # Build with MKL support.
        --config=mkl_aarch64    # Build with oneDNN and Compute Library for the Arm Architecture (ACL).
        --config=monolithic     # Config for mostly static monolithic build.
        --config=numa           # Build with NUMA support.
        --config=dynamic_kernels        # (Experimental) Build kernels into separate shared objects.
        --config=v1             # Build with TensorFlow 1 API instead of TF 2 API.
Preconfigured Bazel build configs to DISABLE default on features:
        --config=nogcp          # Disable GCP support.
        --config=nonccl         # Disable NVIDIA NCCL support.
```

### Building

Finally, we need to build Tensorflow and run some tests. Tensorflow is a HUGE framework, and building it ALL from scratch may not be your best bet, as it will take a lot of time and doesn't really make sense, as we will not modify every part of Tensorflow at once.

Instead, we are going to track down where the compression headers that we saw before are tested. This boils down to the path `tensorflow/core/data`. However, we don't need to run every test from this folder, just the ones needed for compression. These tests are defined in the `tensorflow/core/data/BUILD` file. Specifically, we just need the build target `:compression_utils_tests`, that runs the tests for the target `:compression_utils`.

Before building, we need to install `jdk11`, which is needed by the tests that we are going to run:

```
sudo apt update
sudo apt install default-jdk
```

At the time of writing this, after installing `default-jdk`, the command `javac --version` should output:

```
$ javac --version

javac 11.0.11
```

Here is where `bazel` comes into play. We need to specify the path and build target for what we want to build and run tests for as follows:

```
bazel test //tensorflow/core/data:compression_utils_test
```

Now relax, watch something on YouTube and wait. It will take some time.

At the end of the compilation, you should see something like:

```
INFO: Analyzed target //tensorflow/core/data:compression_utils_test (156 packages loaded, 6776 targets configured).
INFO: Found 1 test target...
Target //tensorflow/core/data:compression_utils_test up-to-date:
  bazel-bin/tensorflow/core/data/compression_utils_test
INFO: Elapsed time: 677.754s, Critical Path: 59.27s
INFO: 1940 processes: 482 internal, 1458 local.
INFO: Build completed successfully, 1940 total actions
//tensorflow/core/data:compression_utils_test                            PASSED in 0.2s

Executed 1 out of 1 test: 1 test passes.
INFO: Build completed successfully, 1940 total actions
```

Success!

## Ehm...

After some more fiddling around, the file `tensorflow/python/lib/io/tf_record_test.py` has some nice usage of the class `tf_record.TFRecordOptions` defined in `tensorflow/python/lib/io/tf_record.py:43`. An instance of this class will actually pass down to `TFRecordWriter` the compression level, method, strategy and other variables:

```
Args:
    compression_type: `"GZIP"`, `"ZLIB"`, or `""` (no compression).
    flush_mode: flush mode or `None`, Default: Z_NO_FLUSH.
    input_buffer_size: int or `None`.
    output_buffer_size: int or `None`.
    window_bits: int or `None`.
    compression_level: 0 to 9, or `None`.
    compression_method: compression method or `None`.
    mem_level: 1 to 9, or `None`.
    compression_strategy: strategy or `None`. Default: Z_DEFAULT_STRATEGY.
```

It can do this because the `ZLibCompressionOptions` class defined in `tensorflow/core/lib/io/zlib_compression_options.h` has the properties automatically mapped here: `tensorflow/python/lib/io/record_io_wrapper.cc:268`

And it would become something like (following test example in `tensorflow/python/lib/io/tf_record_test.py:270`):

```python
import tensorflow as tf
from tensorflow.python.lib.io import tf_record
import zlib

options = tf_record.TFRecordOptions(compression_type=tf_record.TFRecordCompressionType.ZLIB, 
    compression_level=2,
    flush_mode=zlib.Z_NO_FLUSH,
    input_buffer_size=4096, # default is 8192 (8KB)
    output_buffer_size=4096, # default is 8192 (8KB)
    window_bits=8, # default is 15 (leading to 32KB search buffer and 64KB sliding window), see https://github.com/madler/zlib/blob/master/deflate.h#L48 and https://www.euccas.me/zlib/#zlib_sliding_window
    compression_strategy=Z_HUFFMAN_ONLY, # default is Z_DEFAULT_STRATEGY (see https://www.zlib.net/manual.html#Usage, compression_strategy)
    )
tf.io.TFRecordWriter("somefile.tfrecord", options=options)
```

Regarding `compression_strategy`, taken from the [manual](https://www.zlib.net/manual.html):

> The strategy parameter is used to tune the compression algorithm. Use the value Z_DEFAULT_STRATEGY for normal data, Z_FILTERED for data produced by a filter (or predictor), Z_HUFFMAN_ONLY to force Huffman encoding only (no string match), or Z_RLE to limit match distances to one (run-length encoding). Filtered data consists mostly of small values with a somewhat random distribution. In this case, the compression algorithm is tuned to compress them better. The effect of Z_FILTERED is to force more Huffman coding and less string matching; it is somewhat intermediate between Z_DEFAULT_STRATEGY and Z_HUFFMAN_ONLY. Z_RLE is designed to be almost as fast as Z_HUFFMAN_ONLY, but give better compression for PNG image data. The strategy parameter only affects the compression ratio but not the correctness of the compressed output even if it is not set appropriately. Z_FIXED prevents the use of dynamic Huffman codes, allowing for a simpler decoder for special applications.

# 8/11/2021

After finding out that changing ZLIB parameters was going to be extremely easy using just `tf.io.TFRecordOptions`, I implemented the change basing [in a fork repo](https://gitlab.i13.in.tum.de/adrian_castro/presto/).

Before actually testing out the changes however, I needed to have a (much) larger disk to play with, as the datasets themselves are rather big (> 100GB), which I have done by launching the following commands:

```bash
sudo git clone https://github.com/SFTtech/ceph-mount /opt/ceph-mount
sudo ln -s /opt/ceph-mount/mount.ceph /sbin/mount.ceph
sudo mkdir /etc/ceph
 sudo echo "<NILMKEYRING>" > /etc/ceph/nilm.keyring
 sudo echo -e "[global]\nmon_host=mon1.ceph.rbg.tum.de,mon2.ceph.rbg.tum.de,tum.de,mon4.ceph.rbg.tum.de" > /etc/ceph/ceph.conf
mkdir /home/ubuntu/rbgstorage
sudo mount -t ceph :/I13/nilm /home/ubuntu/rbgstorage -o name=I13.fs.nilm,secretfile=/etc/ceph/nilm.keyring
```

(Note: `<NILMKEYRING>` contains the token used by `ceph-mount` to access the volume).

The volume has 10Gb/s up and downlink, which are similar speeds to using a local SSD. Yey. The folder structure of this volume is as follows:

```
.
├── commonvoice
├── cubeplusplus
├── imagenet
├── librispeech
├── openwebtext
├── nilm
├─ word2vec
└── temp
    ├── buchberger
    ├── isenko
    └── unknown
```

We are just interested in the dataset folders (for which we only should read from!), and the `temp` folder. In the `temp` folder we need to create a subdirectory with our name, like `adrian` or `adrian-castro` (I went with the latter, cause yes).

After modifying the `presto` code to adapt to the logs and adding the custom, we need to verify that our modifications yield the same results as the original runs if run with default parameters.

In our case, we made the following additions:

* `presto/strategy.py`: added `compression_level` and `compression_strategy` as init parameters, and added to the logs
* `*_demo.py`: modified to also take `compression_level` and `compression_strategy` as inputs

# 25/11/2021

## Launching benchmarks

As a start, we ran benchmarks for the Cube++ PNG dataset with the following settings:

| Compression | Compression Level | Compression Strategy |
| :--: | :--: | :--: |
| None | - | - |
| ZLIB | 6 | Z_DEFAULT_STRATEGY |
| ZLIB | 6 | Z_FILTERED |
| ZLIB | 6 | Z_RLE |
| ZLIB | 6 | Z_HUFFMAN_ONLY |
| ZLIB | 6 | Z_FIXED |

We started like this just to test if the code that we just wrote was actually changing some results, and in fact, different compression strategies indeed change the results with respect to no compression, or also across different compression strategies.

<!-- FIXME: add benchmark results for Cube++ PNG -->

With the results at hand, we can replicate the same approach for the other datasets, so that we have a better idea of which compression strategy works best or worst for which type of dataset.

## Doing stuff in the meantime

Because benchmarks run for so long (even days), we can start to explore how to implement different compression algorithms and libraries in Tensorflow.

Recently (24/11/2021) a [brand-new compression algorithm popped up](https://lobste.rs/s/1hafjp/lossless_image_compression_o_n_time) called QOI (Quite OK Image), a format for fast, lossless image compression. This gives us a nice idea about using specific file types, optimized for compression.

# 09/12/2021

I have [presented the findings of the benchmarks](https://docs.google.com/presentation/d/19W38GXh3ujM4fDGHZYQAgaJaHGHigM0Qx5UlhqeWYvU/) that we have been running over the past days (although some are still running to this day, namely `openwebtext`).

Some more papers I have went through showcasing comparison of different compression algorithms.

**TLDR**:

- Double floating point data
    - pFPC
    - SPDP
- General purpose
    - zstd
    - zlib
    - LZMA

## Floating Point Data

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4015488

Old paper (2006), nice comparison of different compression algorithms for FP data, including ZLIB

https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4015488

FPcrush, a tool that automatically synthesizes an optimized compressor for each given input. The synthesized algorithms are lossless and parallelized using OpenMP.

Much more recent paper (2016, introduced here: https://kubework.slack.com/archives/D02A3GM8RBP/p1638227111006700) showcasing different compression algorithms with parallel computation support. Even though the paper does not offer any source code, and just the binary of their tool, their findings with respect to FP data compression with the shown algorithms can be quite useful.

For example, the only case where the compression ratio went over 1.6 was on a dataset where the percent of unique values was of 0.3 (a lot of repeated values). Although, compression and decompression speeds could still be taken into consideration for our purposes? pFPC performs very well in that regard, and its [source code is available](https://userweb.cs.txstate.edu/~burtscher/research/pFPC/).

https://userweb.cs.txstate.edu/~mb92/papers/dcc18.pdf

SPDP. Same author as FPcrush and pFPC. Source code available [here](https://userweb.cs.txstate.edu/~burtscher/research/SPDPcompressor/). The interesting thing is that they showcase the use of zstd , an algorithm developed by Facebook and with [source code available](https://github.com/facebook/zstd).

## General Purpose

https://github.com/facebook/zstd

https://www.zlib.net/

[LZMA (used by 7-Zip)](https://www.7-zip.org/sdk.html)

# 23/01/2022

Over the past month, we have successfully implemented the [ZSTD](https://github.com/facebook/zstd) library in TensorFlow, and [created a PR for it](https://github.com/tensorflow/tensorflow/pull/53385). Even though it is based upon an old release `r2.6`, it serves our purposes.

In particular, we have used the Streaming Compression API. This is because we integrated ZSTD into [`tf.io.TFRecordOptions`](https://www.tensorflow.org/api_docs/python/tf/io/TFRecordOptions)'s class, which is fed into `TFRecordWriter` and writes blocks of data in streaming fashion, meaning that we cannot know in advance input size, which is a requirement for regular ZSTD compression.

We have left compression parameters to default:
- `window_log`: 0
- `compression_level`: 3 (`ZSTD_CLEVEL_DEFAULT`)
- `compression_strategy`: 0
- `nb_workers`: 0 (single worker mode)

## Testing

Using ZSTD against the same datasets and with the default parameters stated before, shows interesting results. In this section, we will show different graphs created using data from several runs of `Presto` on the `Cube++-PNG` dataset. In these sections, all the compression type options that are shown were run 5 times each.

### Dstat metrics over time

![a](images/zstd_cached_memory_in_mb.png)
![b](images/zstd_network_read_in_mbs.png)
![c](images/zstd_network_write_in_mbs.png)
![d](images/zstd_cpu_usr_in_percent.png)

Compression using GZIP and ZLIB both result in slower compression, seemingly depending on the fact that both of these methods seem to read very slowly, and write data to disk just as slowly. This looks a bit strange, and may be due to problems in the implementation of the compression algorithm.

Our ZSTD implementation on the other hand seems to be reading data from disk quite fast, and write compressed data to disk at nearly the speed of no compression at all. This alone is a huge step, and we will see why in the next sections.

### Throughput Samples per Second

ZSTD however did not show any significant improvements with respect to TSPS, in fact, it performed worse than GZIP and ZLIB.

![c](images/zstd_tsps_graph_line.png)

When comparing absolute times for each step of the pipeline, the difference between different implementations is way more significant. ZSTD only adds around 3 more minutes to the total time 

![c](images/zstd_offline_online_time.png)

### Final Thoughts

The inconsistency of reproducibility results of using CEPH is a problem here. We want to re-run our experiments on local storage, and see what happens.
