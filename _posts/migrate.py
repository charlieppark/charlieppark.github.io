from pathlib import Path

old_path = './old'
new_path = './new'

for subpath in Path(old_path).iterdir():
    for file in subpath.iterdir():
        if file.suffix == '.md':
            filename = str(file)
    
    with open(filename, 'r') as fr:
        lines = fr.readlines()
    count = 0
    newlines = []
    for line in lines:
        if line.startswith('---'):
            if count == 0:
                newlines.append('---')
                count += 1
                continue

            elif count == 1:
                newlines.append('toc: false')
                newlines.append('comments: true')
                newlines.append('math: true')
                newlines.append('mermaid: false')
                newlines.append('# img_path: /img/path/')
                newlines.append('pin: false')
                title = ' '.join([title, subtitle])
                newlines.append('title: ' + title)
                newlines.append('---')
                count = -1
                continue

            else:
                continue

        if line.startswith('layout:') or \
        line.startswith('type:') or \
        line.startswith('post-header:') or \
        line.startswith('draft:') or \
        line.startswith('header-img:'):
            continue

        if line.startswith('date: '):
            date = line.split(' ')[1]

        if line.startswith('hash-tag:'):
            line = line.replace('hash-tag:', 'tag:')

        if line.startswith('subtitle:'):
            subtitle = ' '.join(line.split()[1:])
            continue
        
        if line.startswith('writer:'):
            line = line.replace('writer:', 'author:')
        
        if line.startswith('category:'):
            line = line.replace('category:', 'categories:')
        
        if line.startswith('title:'):
            title = ' '.join(line.split()[1:])
            newfilename = (date + '-' + title + '.md').replace(' ', '_').replace('/', '_')
            continue

        newlines.append(line.replace('\n', '').replace('\r', ''))

    with open(new_path + '/' + newfilename, 'w') as fw:
        fw.write('\n'.join(newlines))