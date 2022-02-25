# Image Processing Tools:

<table>
<tr>
<td>Base image</td>
<td>Interactive Complex as shown below</td>
</tr>
<tr>
<td>
<img src="/TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg">
</td>
<td>
<img src="/TestFiles/COMPLEX/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg">
</td>
</tr>
</table>

## Test images

* [TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg](
  /TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg)  Photo by
  <a href="https://unsplash.com/@aranprime?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
  aranprime</a>
  on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
  Unsplash</a>
* [TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg](
  /TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg) Photo by
  <a href="https://unsplash.com/@coryb?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
  Cory Bouthillette</a>
  on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">
  Unsplash</a>

## Install

Install requirements with pip \
`python -m pip install -r requirements.txt`

## Commandline Arguments:

```commandline 
  -h, --help            show this help message and exit
  -d DIRECTORY, --directory DIRECTORY
                        directory of files to apply methods to
  -f FUNCTION, --function FUNCTION
                        functions to apply over each file 'add area' use ',' as separator or 'all'
  --allow_sub_folders, --no-allow_sub_folders
                        run function on all sub-folders of the given directory (default: False)
  -m, --multicore, --no-multicore
                        use multiple CPU cores (default: False)
  -s SCALE_FACTOR, --scale SCALE_FACTOR
                        Scale factor to use, 2 for double size, 0.5 for half size, etc
  -t, --timer, --no-timer
                        print timer stats -- Not usable with multicore -- (default: False)
  -fn, --filename, --no-filename
                        modify filename for output file instead of folder (default: False)

```
