import torch
num_users=1892
num_items=4489

def computer(users_emb,items_emb,Graph):
    """
    propagate methods for lightGCN
    """
    #users_emb = self.embedding_user.weight  # (1892,64)
    #items_emb = self.embedding_item.weight  # (4489,64)
    with torch.no_grad():
        all_emb = torch.cat([users_emb, items_emb])  # (6381,64)
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        g_droped = Graph
        for layer in range(3):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_users, num_items])
        return users, items