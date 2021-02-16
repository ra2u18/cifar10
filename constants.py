
config = {
    'train_path': './output/train',
    'valid_path': './output/val',
    'test_path': './testing',
    'checkpoints': './checkpoints'
}

classes = ['bedroom', # 0
    'coast', # 1
    'forest', # 2
    'highway', # 3
    'industrial', # 4
    'insidecity',# 5
    'kitchen', # 6
    'living-room',# 7           
    'mountain', # 8
    'office', #9            
    'open-country',#10      
    'store', # 11
    'street',#12    
    'suburb',#13          
    'tall-building', #14
]

batch_size = 8
resized_output = (200, 200)
num_classes= 15

SEED = 42