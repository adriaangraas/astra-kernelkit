kernelkit.torch\_support.AutogradOperator
=========================================

.. currentmodule:: kernelkit.torch_support

.. autoclass:: AutogradOperator

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~AutogradOperator.__init__
      ~AutogradOperator.apply
      ~AutogradOperator.backward
      ~AutogradOperator.forward
      ~AutogradOperator.jvp
      ~AutogradOperator.mark_dirty
      ~AutogradOperator.mark_non_differentiable
      ~AutogradOperator.mark_shared_storage
      ~AutogradOperator.maybe_clear_saved_tensors
      ~AutogradOperator.name
      ~AutogradOperator.register_hook
      ~AutogradOperator.register_prehook
      ~AutogradOperator.save_for_backward
      ~AutogradOperator.save_for_forward
      ~AutogradOperator.set_materialize_grads
      ~AutogradOperator.setup_context
      ~AutogradOperator.vjp
      ~AutogradOperator.vmap
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~AutogradOperator.dirty_tensors
      ~AutogradOperator.generate_vmap_rule
      ~AutogradOperator.materialize_grads
      ~AutogradOperator.metadata
      ~AutogradOperator.needs_input_grad
      ~AutogradOperator.next_functions
      ~AutogradOperator.non_differentiable
      ~AutogradOperator.requires_grad
      ~AutogradOperator.saved_for_forward
      ~AutogradOperator.saved_tensors
      ~AutogradOperator.saved_variables
      ~AutogradOperator.to_save
   
   