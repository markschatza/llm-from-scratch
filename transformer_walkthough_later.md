Later need to walkthough the following

In our real code, the backward path starts from the scalar loss and walks back through everything that produced it.
In pretrain/transformer.py, the forward pass is roughly:
python



logits, loss = model(x, y)

The computation path is:
text



token IDs x
-> token embeddings
-> position embeddings
-> add token + position vectors
-> Transformer blocks
-> final layer norm
-> lm_head
-> logits
-> cross entropy loss against y

When we call:
python



loss.backward()

PyTorch walks that graph backward:
text



loss
-> cross entropy
-> logits
-> lm_head
-> final layer norm
-> Transformer block 2
-> Transformer block 1
-> token embeddings + position embeddings

Inside each Transformer block, it walks backward through two main branches:
text



block output
-> residual add after feed-forward
-> feed-forward network
-> layer norm before feed-forward
-> residual add after attention
-> causal self-attention
-> layer norm before attention

The feed-forward branch is:
text



ffwd output
-> dropout
-> second linear layer
-> GELU activation
-> first linear layer

So gradients get computed for:
text



ffwd.net[0].weight
ffwd.net[0].bias
ffwd.net[2].weight
ffwd.net[2].bias

The attention branch is more involved:
text



attention output
-> output projection linear layer
-> attention weights @ values
-> softmax
-> causal mask
-> query @ key transpose
-> q, k, v split
-> qkv linear layer

So gradients get computed for:
text



attn.qkv.weight
attn.qkv.bias
attn.proj.weight
attn.proj.bias

The causal mask affects which positions can attend to which earlier positions, but the mask itself is not learned. It does not get updated.
Then after all blocks, gradients keep flowing back to:
text



token_embedding.weight
position_embedding.weight

So the full backward path is basically:
text



loss
-> prediction head
-> final normalization
-> transformer blocks in reverse order
-> attention + feed-forward internals
-> input embeddings

Important detail: residual connections split and merge gradient flow.
When the forward pass does:
python



x = x + self.attn(self.ln1(x))

the backward pass sends gradient through both paths:
text



path 1: directly back through x
path 2: back through attention/layernorm

That is one reason Transformers train well: gradients have shorter paths backward through residual connections, instead of being forced through every transformation sequentially.