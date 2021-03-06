# Image Processing Tools:

<table>
<tr>
<td>Base image</td>
<td>Interactive Complex as shown below</td>
</tr>
<tr>
<td>
<img src="https://github.com/Quiet-Clicking-Sounds/ImageProcessingTools/blob/main//TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg">
</td>
<td>
<img src="https://github.com/Quiet-Clicking-Sounds/ImageProcessingTools/blob/main//TestFiles/COMPLEX/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg">
</td>
</tr>
</table>

## Test images

* [TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg](TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg)  Photo by <a href="https://unsplash.com/@aranprime?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">aranprime</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
* [TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg](TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg) Photo by <a href="https://unsplash.com/@coryb?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Cory Bouthillette</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>

## Install

Install requirements with pip \
`python -m pip install -r requirements.txt`


## Usage Examples

use `-h` or `--help` to access commandline help.

### Single pass via commandline

`ImageProcessingTools.py -f TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg -o TestFiles/Multi/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg -w 5`

### Multi pass via commandline

`ImageProcessingTools.py -f TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg -o TestFiles/Multi/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg -w 3,5,7,13,19 -combine sum`



### Interactive

```
ImageProcessingTools.py -i
Target input file: TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg
Target output file: TestFiles/INTERACTIVE/aranprime-Wa6KJdX2Sy8-unsplash.jpg
Window size, int or list[int]: 3,5,7,13,19
Additional arguments, --no-rgb: 
--------------------
Beginning operation
method used to combine multi-pass images: 'sum', 'avg', 'dist' - prepend '-' to invert list: -dist
Operation Complete
--------------------
run again on same file? y/n: n
```

### Interactive Complex
Interactive complex allows for multiple single or multi pass methods to be layered together.  

```
ImageProcessingTools.py -ic
Target input file: TestFiles/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg
Target output file: TestFiles/COMEPLEX/cory-bouthillette-nop6Tqlt-DE-unsplash.jpg
Additional arguments, --no-rgb: 
--------------------
Window syntax: [int] or [list[int] combine] e.g. '5' or '3,5,7 dist'
combine options: sum, avg, dist, -dist 
Use 'q' or 'quit' to exit window input

Window 0: 3,5,7,9,11,13,15 dist
Window 1: 3,5,7,9,11,13,15 -dist
Window 2: 3,5,7 avg
Window 3: 3,5 avg
Window 4: 25
Window 5: q
Final combination method: sum 
```





