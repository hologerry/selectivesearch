import pickle

# detectron_proposal_file = '/D_data/Self/data/coco/proposals/coco_2017_train_box_proposals_21bc3a.pkl'
detectron_proposal_file = '/D_data/Self/data/coco/proposals/coco_2017_val_box_proposals_ee0dad.pkl'


with open(detectron_proposal_file, 'rb') as f:
    proposals = pickle.load(f)

print(type(proposals))
print("keys", proposals.keys())
