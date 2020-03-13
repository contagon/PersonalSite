title: Jupyter to Flask
date: 2020-03-12
description: How I convert Jupyter Notebooks to use in Flask
tags:
  - programming

When searching for how to put my site together, using Jupyter notebooks directly was a necessity. I use them in class projects, personal projects, etc, etc and being able to use them directly would be a huge productivity boost.

Nikola was a great option, but seemed to be more blog oriented, and had way more configuration than what I was looking for. Jupyter's own `nbconvert` was also an option, but either seem to come with WAY too much CSS, or not enough.

I landed upon using `nbconvert`, but with the markdown option instead. This way it converts directly to what the rest of my posts look like, and flask can treat them like all the others. Everytime my site is built it "reconverts" the notebooks right before flask freezes the site.

Unfortunately, `nbconvert` doesn't convert quite right... but was close enough a little code fixes handled the rest. I'll walk you through what this process looks like.

One of the large reasons I love Jupyter notebooks is the inline latex available in the markdown. To get these running in my site, we use MathJax javascript library. Including
```html
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$']]}
});
</script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML"></script>
```
in the header does the trick. I further enable dollar signs to activate math mode since I don't use dollar signs anywhere else.

We convert the file itself, where `type` refers to project/post and `file` is the name of the file.. fittingly!
```python
os.system(f'jupyter nbconvert --to markdown content/{type}/{file}')
```

Next, we clear all previous images made by the notebook, and replace them with the images just made. Still haven't decided if this will be the final state of how my images on stored however.
```python
filename = file.split(".")[0]
files = fr'{filename}_files'
os.system(f'rm -rf static/{type}/{files}')
os.system(f'mv content/{type}/{files} static/{type}')
```

Next, the markdown itself seemed to be missing various things, including  
  -  `\\` endline characters in Latex needs to be escaped and becomes `\\\\\\\\`  
  -  The underscore in `\_{` in Latex also needs to be escaped and becomes `\_{`  
  -  Image location needs to be moved to `/static/{type}/{files}`  
  -  I added a `#` to each heading to make formatting match up with the rest of the site  

The code for all these replacements was done using regex and ends up looking like:
```python
with open(f'content/{type}/{filename}.md', 'r') as reader:
      md = reader.read()
  md = re.sub(r'\\\\', r'\\\\\\\\', md)
  md = re.sub(r'_{', '\_{', md)
  md = re.sub(files, f'/static/{type}/{files}', md)
  md = re.sub(r'(#+)', r'#\1', md)
  with open(f'content/{type}/{filename}.md', 'w') as writer:
      writer.write(md)
```

And after all that frozen flask handles the rest! You can see the code in it's natural environment [here](https://github.com/contagon/PersonalSite/blob/master/startup.py)

If you're too lazy to check out the repo, it ends up as follows:

```python
def convert_jupyter(type, file):
    #do initial conversions
    os.system(f'jupyter nbconvert --to markdown content/{type}/{file}')

    #remove previous and move data to static
    filename = file.split(".")[0]
    files = fr'{filename}_files'
    os.system(f'rm -rf static/{type}/{files}')
    os.system(f'mv content/{type}/{files} static/{type}')

    #open and replace various things so it works
    with open(f'content/{type}/{filename}.md', 'r') as reader:
        md = reader.read()
    md = re.sub(r'\\\\', r'\\\\\\\\', md)
    md = re.sub(r'_{', '\_{', md)
    md = re.sub(files, f'/static/{type}/{files}', md)
    md = re.sub(r'(#+)', r'#\1', md)
    with open(f'content/{type}/{filename}.md', 'w') as writer:
        writer.write(md)

def convert_all():
    #convert all posts
    for filename in os.listdir("content/posts/"):
        if filename.endswith(".ipynb"):
            convert_jupyter('posts', filename)

    #convert all projects
    for filename in os.listdir("content/projects/"):
        if filename.endswith(".ipynb"):
            convert_jupyter('projects', filename)
```
