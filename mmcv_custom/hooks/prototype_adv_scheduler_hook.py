import warnings
from typing import Optional

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class PrototypeAdversarialSchedulerHook(Hook):
    """Activate prototype adversarial loss after a given epoch.

    Args:
        activate_epoch (int): 1-based epoch index when the adversarial loss
            should start contributing to the total loss.
        target_adv_weight (float, optional): The loss weight to set once
            activated. If ``None``, the existing weight is kept.
        target_grl_weight (float, optional): Desired gradient reversal
            coefficient after activation. If ``None``, no change.
        target_momentum (float, optional): Momentum value to assign to the
            prototype bank after activation. If ``None``, no change.
        verbose (bool): Whether to log weight changes.
    """

    def __init__(
        self,
        activate_epoch: int = 11,
        target_adv_weight: Optional[float] = None,
        target_grl_weight: Optional[float] = None,
        target_momentum: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        if activate_epoch <= 0:
            raise ValueError('activate_epoch must be positive (1-based).')
        self.activate_epoch = activate_epoch
        self.target_adv_weight = target_adv_weight
        self.target_grl_weight = target_grl_weight
        self.target_momentum = target_momentum
        self.verbose = verbose
        self._has_applied = False

    def after_train_epoch(self, runner) -> None:
        if self._has_applied:
            return
        current_epoch = runner.epoch + 1  # convert to 1-based numbering
        if current_epoch < self.activate_epoch:
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        manager = getattr(model, 'prototype_manager', None)
        if manager is None:
            warnings.warn('Prototype manager not found when applying '
                          'PrototypeAdversarialSchedulerHook.', RuntimeWarning)
            self._has_applied = True
            return

        if self.target_adv_weight is not None:
            manager.adv_weight = float(self.target_adv_weight)

        if self.target_grl_weight is not None:
            if hasattr(manager, 'set_gradient_reversal_weight'):
                manager.set_gradient_reversal_weight(float(self.target_grl_weight))
            elif hasattr(manager, 'adv_loss'):
                adv_loss = manager.adv_loss
                if adv_loss is not None and hasattr(adv_loss, 'gradient_reversal'):
                    adv_loss.gradient_reversal.lambd = float(self.target_grl_weight)

        if self.target_momentum is not None:
            if hasattr(manager, 'bank'):
                manager.bank.momentum = float(self.target_momentum)

        if self.verbose and hasattr(runner, 'logger') and runner.logger is not None:
            grl_module = getattr(getattr(manager, 'adv_loss', None), 'gradient_reversal', None)
            grl_value = getattr(grl_module, 'lambd', None)
            momentum_value = getattr(getattr(manager, 'bank', None), 'momentum', None)
            runner.logger.info(
                'Prototype adversarial scheduler activated at epoch %d: '
                'adv_weight=%s, grl_weight=%s, momentum=%s',
                current_epoch,
                getattr(manager, 'adv_weight', None),
                grl_value,
                momentum_value,
            )

        self._has_applied = True
