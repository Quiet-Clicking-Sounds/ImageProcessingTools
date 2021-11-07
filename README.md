<h1>Image Processing Tools:</h1>

<h2> Usage Examples</h2>

<h3> Single pass via commandline</h3>

`ImageProcessingTools.py -f TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg -o TestFiles/Multi/aranprime-Wa6KJdX2Sy8-unsplash.jpg -w 5`

<h3> Multi pass via commandline</h3>

`ImageProcessingTools.py -f TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg -o TestFiles/Multi/aranprime-Wa6KJdX2Sy8-unsplash.jpg -w 3,5,7,13,19 -combine sum`



<h3> Interactive</h3>

```
ImageProcessingTools.py -i\
Target input file: TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg\
Target output file: TestFiles/INTERACTIVE/aranprime-Wa6KJdX2Sy8-unsplash.jpg\
Window size, int or list[int]: 3,5,7,13,19\
Additional arguments, --no-rgb: \
--------------------\
Beginning operation\
method used to combine multi-pass images: 'sum', 'avg', 'dist' - prepend '-' to invert list: -dist\
Operation Complete\
--------------------\
run again on same file? y/n: n\
```

<h3>Interactive Complex</h3>

```
ImageProcessingTools.py -ic\
Target input file: TestFiles/graham-mansfield-TTnHA_4UzmA-unsplash.jpg\
Target output file: TestFiles/COMEPLEX/graham-mansfield-TTnHA_4UzmA-unsplash.jpg\
Additional arguments, --no-rgb: \
--------------------\
Window syntax: [int] or [list[int] combine] e.g. '5' or '3,5,7 dist'\
combine options: sum, avg, dist, -dist \
Use 'q' or 'quit' to exit window input\
\
Window 0: 3,5,7,9,11,13,15 dist\
Window 1: 3,5,7,9,11,13,15 -dist\
Window 2: 3,5,7 avg\
Window 3: 3,5 avg\
Window 4: 25\
Window 5: q\
`Final combination method: sum 
```



<hr> 
<h2>Test images</h2>

- TestFiles/graham-mansfield-TTnHA_4UzmA-unsplash.jpg Photo by <a href="https://unsplash.com/@grahammansfield1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">GRAHAM MANSFIELD</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- TestFiles/dave-goudreau-PpQt_OGg9ZA-unsplash.jpg  Photo by <a href="https://unsplash.com/@davegoudreau?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Dave Goudreau</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- TestFiles/aranprime-Wa6KJdX2Sy8-unsplash.jpg  Photo by <a href="https://unsplash.com/@aranprime?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">aranprime</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
- TestFiles/devon-divine-GMIaeFdtXMA-unsplash.jpg  Photo by <a href="https://unsplash.com/@lightrisephoto?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Devon Divine</a> on <a href="https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
