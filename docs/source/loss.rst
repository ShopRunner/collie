Losses
======

**Standard Implicit Loss Functions**

A Collie model can't train without a loss function, and Collie comes out-of-the-box with two different standard loss function calculations: **Bayesian Personalized Ranking (BPR) loss** and **Hinge loss**.

In its simplest form, each loss function accepts as input a prediction score for a positive item (an item a user has interacted with), and a prediction score for a negative item (an item a user has not interacted with). While the mathematical details differ between each loss, generally, **all implicit losses will punish a model for ranking a negative item higher than a positive item**. The severity of this punishment differs across loss functions, as shown in the example below.

.. code-block:: python

   import torch

   from collie_recs.loss import bpr_loss, hinge_loss


   # an ideal loss case
   positive_score = torch.tensor([3.0])
   negative_score = torch.tensor([1.5])

   print('BPR Loss:  ', bpr_loss(positive_score, negative_score))
   print('Hinge Loss:', hinge_loss(positive_score, negative_score))

   print('\n-----\n')

   # a less-than-ideal loss case
   positive_score = torch.tensor([1.5])
   negative_score = torch.tensor([3.0])

   print('BPR Loss:  ', bpr_loss(positive_score, negative_score))
   print('Hinge Loss:', hinge_loss(positive_score, negative_score))


.. code-block:: bash

   BPR Loss:   tensor(0.2157)
   Hinge Loss: tensor(0.)

   -----

   BPR Loss:   tensor(1.4860)
   Hinge Loss: tensor(8.7500)

**Adaptive Implicit Loss Functions**

Some losses extend this idea by being "adaptive," or accepting multiple negative item prediction scores for each positive score. These losses are typically much more punishing than "non-adaptive" losses, since they allow more opportunities for the model to incorrectly rank a negative item higher than a positive one. These losses include **Adaptive Bayesian Personalized Ranking loss**, **Adaptive Hinge loss**, and **Weighted Approximately Ranked Pairwise (WARP) loss**.

.. code-block:: python

   import torch

   from collie_recs.loss import adaptive_bpr_loss, adaptive_hinge_loss, warp_loss


   # an ideal loss case
   positive_score = torch.tensor([3.0])
   many_negative_scores = torch.tensor([[1.5], [0.5], [1.0]])

   print('Adaptive BPR Loss:  ', adaptive_bpr_loss(positive_score, many_negative_scores))
   print('Adaptive Hinge Loss:', adaptive_hinge_loss(positive_score, many_negative_scores))
   print('WARP Loss:          ', warp_loss(positive_score, many_negative_scores, num_items=3))

   print('\n-----\n')

   # a less-than-ideal loss case
   positive_score = torch.tensor([1.5])
   many_negative_scores = torch.tensor([[2.0], [3.0], [2.5]])

   print('Adaptive BPR Loss:  ', adaptive_bpr_loss(positive_score, many_negative_scores))
   print('Adaptive Hinge Loss:', adaptive_hinge_loss(positive_score, many_negative_scores))
   print('WARP Loss:          ', warp_loss(positive_score, many_negative_scores, num_items=3))
   print('WARP Loss:          ', warp_loss(positive_score, many_negative_scores, num_items=30))

   print('\n-----\n')

   # a case where multiple negative items gives us greater opportunity to correct the model
   positive_score = torch.tensor([1.5])
   many_negative_scores = torch.tensor([[1.0], [4.0], [1.49]])

   print('Adaptive BPR Loss:  ', adaptive_bpr_loss(positive_score, many_negative_scores))
   print('Adaptive Hinge Loss:', adaptive_hinge_loss(positive_score, many_negative_scores))
   print('WARP Loss:          ', warp_loss(positive_score, many_negative_scores, num_items=3))
   print('WARP Loss:          ', warp_loss(positive_score, many_negative_scores, num_items=30))

.. code-block:: bash

    Adaptive BPR Loss:   tensor(0.2157)
    Adaptive Hinge Loss: tensor(0.)
    WARP Loss:           tensor(0.)

    -----

    Adaptive BPR Loss:   tensor(1.4860)
    Adaptive Hinge Loss: tensor(8.7500)
    WARP Loss:           tensor(4.3636)
    WARP Loss:           tensor(31.1301)

    -----

    Adaptive BPR Loss:   tensor(1.7782)
    Adaptive Hinge Loss: tensor(15.7500)
    WARP Loss:           tensor(0.8510)
    WARP Loss:           tensor(4.5926)


**Partial Credit Loss Functions**

If you have item metadata available, you might reason that not all losses should be equal. For example, say you are training a recommendation system on MovieLens data, where users interact with different films, and you are comparing a positive item, *Star Wars*, with two negative items: *Star Trek* and *Legally Blonde*.

Normally, the loss for *Star Wars* compared with *Star Trek*, and *Star Wars* compared with *Legally Blonde* would be equal. But, as humans, we know that *Star Trek* is closer to *Star Wars* (both being space western films) than *Legally Blonde* is (a romantic comedy that does not have space elements), and would want our loss function to account for that [#f1]_.

For these scenarios, all loss functions in Collie support partial credit calculations, meaning we can provide metadata to reduce the potential loss for certain items with matching metadata. This is best seen through an example below:

.. code-block:: python

   import torch

   # we'll just look at ``bpr_loss`` for this, but note that this works with
   # all loss functions in Collie
   from collie_recs.loss import bpr_loss


   # positive item is Star Wars
   star_wars_score = torch.tensor([1.0])

   # negative items are Star Trek and Legally Blonde
   star_trek_score = torch.tensor([3.0])
   legally_blonde_score = torch.tensor([3.0])

   print('Star Wars vs Star Trek Loss:      ', end='')
   print(bpr_loss(positive_scores=star_wars_score, negative_scores=star_trek_score))

   print('Star Wars vs Legally Blonde Loss: ', end='')
   print(bpr_loss(positive_scores=star_wars_score, negative_scores=legally_blonde_score))

   print('\n-----\n')

   # now let's apply a partial credit calculation to the loss
   metadata_weights = {'genre': 0.25}

   # categorically encode Sci-Fi as ``0`` and Comedy as ``1`` and
   # order values by Star Wars, Star Trek, Legally Blonde
   metadata = {'genre': torch.tensor([0, 0, 1])}

   print('Star Wars vs Star Trek Partial Credit Loss:      ', end='')
   print(bpr_loss(positive_scores=star_wars_score,
                  negative_scores=star_trek_score,
                  positive_items=torch.tensor([0]),
                  negative_items=torch.tensor([1]),
                  metadata=metadata,
                  metadata_weights=metadata_weights))

   print('Star Wars vs Legally Blonde Partial Credit Loss: ', end='')
   print(bpr_loss(positive_scores=star_wars_score,
                  negative_scores=legally_blonde_score,
                  positive_items=torch.tensor([0]),
                  negative_items=torch.tensor([2]),
                  metadata=metadata,
                  metadata_weights=metadata_weights))


.. code-block:: bash

    Star Wars vs Star Trek Loss:      tensor(1.6566)
    Star Wars vs Legally Blonde Loss: tensor(1.6566)

    -----

    Star Wars vs Star Trek Partial Credit Loss:      tensor(1.0287)
    Star Wars vs Legally Blonde Partial Credit Loss: tensor(1.6566)


See :ref:`Tutorials` for a more in-depth example using partial credit loss functions.


Standard Losses
---------------

BPR Loss
^^^^^^^^
.. autofunction:: collie_recs.loss.bpr_loss

Hinge Loss
^^^^^^^^^^
.. autofunction:: collie_recs.loss.hinge_loss

Adaptive Losses
---------------

Adaptive BPR Loss
^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.loss.adaptive_bpr_loss

Adaptive Hinge Loss
^^^^^^^^^^^^^^^^^^^
.. autofunction:: collie_recs.loss.adaptive_hinge_loss

WARP Loss
^^^^^^^^^
.. autofunction:: collie_recs.loss.warp_loss


.. rubric:: Footnotes

.. [#f1] If it were up to the author of this library, everyone would be recommended *Legally Blonde*. It is a fantastic film.
