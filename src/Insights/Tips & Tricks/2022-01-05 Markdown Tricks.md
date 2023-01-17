---
title: Markdown Tricks
date: 2022-04-22 23:58:19
description:
tags: 
 - markdown
 - mkdocs
---

## Jupyter Book
https://jupyterbook.org/en/stable/intro.html
### libraries with Jupyter Book

>jupytext
>mkdocs-jupyter
>mkdocs-awesome-pages-plugin



## Mkdocs
### [Reproducible Reports with MkDocs](https://timvink.nl/reproducible-reports-with-mkdocs/)

### Mkdocs Setup
``` bash
pip install -r requirements.txt # Installs python librarires
mkdocs serve -f mkdocs.yml # Serves docs Folder locally 
mkdocs serve -f mkdocs.yml --dev-addr 0.0.0.0:80 # Serves docs Folder in the local network 
mkdocs build -f mkdocs.yml # Builds docs Site locally 
```

### Mkdocs Plugins

```
https://github.com/mkdocs/mkdocs/wiki/MkDocs-Plugins#pdf--site-conversion
https://github.com/greenape/mknotebooks
https://facelessuser.github.io/pymdown-extensions/extras/slugs/
https://chrieke.medium.com/the-best-mkdocs-plugins-and-customizations-fc820eb19759
```

#### mknotebooks
Add this Css to get pandas dataframe to display properly
```
table {
    display: block;
    max-width: -moz-fit-content;
    max-width: fit-content;
    margin: 0 auto;
    overflow-x: auto;
    white-space: nowrap;
  }
```


#### Admonitions
- warning
- example
- note 
- error
- info
- question     ```

##### Collapsable 
```
??? note annotate "ABS"
	### Try Heading
	- **A** : AA
```

```shell
!!! quote
    Romain Clement’s datasette-dashboards plugin lets you configure dashboards for Datasette using YAML, combining markdown blocks, Vega graphs and single number metrics using a layout powered by CSS grids. This is a beautiful piece of software design, with a very compelling live demo.

    -- <cite>Simon Willison - Creator of Datasette ([source](https://simonwillison.net/2022/Apr/7/datasette-dashboards/))</cite>
```

##### Toggles
```
===  "Client 1"
	A

===  "Client 2"
	B

===  "Client 3"
	C
```

##### Mixture
```
!!! warning "Automatically generated files"
    === "Unordered List"
        ```markdown
	        * Sed sagittis eleifend rutrum       
	    ```
    === "Ordered List"
        ```markdown
	        1. Sed sagittis eleifend rutrum
        ```
```

## Markdown Badges
https://github.com/Ileriayo/markdown-badges/edit/master/README.md

##  Font Awesome
```
[Github Repo] [repository]
[:fontawesome-solid-globe: Live Demo][demo]

[repository]: https://github.com/harsh-maheshwari/harsh-maheshwari "GitHub Repository"
[demo]: https://datasette-dashboards-demo.vercel.app "Live Demo"
```

## FastPages
_powered by [fastpages](https://github.com/fastai/fastpages)_

- [Writing Blogs With Jupyter](https://github.com/fastai/fastpages#writing-blog-posts-with-jupyter)
- [Writing Blogs With Markdown](https://github.com/fastai/fastpages#writing-blog-posts-with-markdown)
- [Writing Blog Posts With Word](https://github.com/fastai/fastpages#writing-blog-posts-with-microsoft-word)



## Links
[Record an Unrecordable Video](https://qr.ae/pGlh7o)
Use Screen Cast on Phone and Screen Video and Microphone Audio Recorder