import re
text = '<p>使用<code>FacetGrid</code>绘制出图形后，有时候我们想设置每个图形的尺寸或者是宽高比，那么我们可以通过在<code>FacetGrid</code>中设置<code>height</code>和<code>aspect</code>来实现，其中<code>height</code>表示的是每个图形的尺寸（默认是宽高一致），<code>aspect</code>表示的是<code>宽度/高度</code>的比例。</p>'
text2 = re.sub(r'<.*?>','',text)
print(text2)


