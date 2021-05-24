import json
import time
import sys
import os
import logging

from web3 import Web3, middleware
from web3.gas_strategies.time_based import construct_time_based_gas_price_strategy

from config import private_key, address

class Prediction:
    
    # bet params
    min_balance_size = 0.1      # min bnb balance
    gas_fee_reserve = 0.05      # bnb reserved for gas fee
    bull_win_rate = 0.50        # bull win rate
    min_prize_pool = 25         # min prize pool size allowed
    min_kelly = 0.01            # min kelly fraction to ensure positive bet size
    balance_override = 0        # balance override (0 = no override)

    # execution params
    execution_block = 4         # transaction is fired when lock block - current block <= execution_block
    gas_price = 5               # default gas price
    gas = 200000                # default gas quantity
    max_wait_seconds = 5        # gas optimizer max wait time
    gas_sample_size = 50        # gas optimizer sample size

    # settings
    logger = logging.getLogger(__name__)
    polling_seconds = 0.5

    def __init__(
        self,
        address,
        private_key,
        bsc="https://bsc-dataseed.binance.org/",
        contract_address='0x516ffd7D1e0Ca40b1879935B2De87cb20Fc1124b',
        gas_optimizer=False
    ):
        self.w3 = Web3(Web3.HTTPProvider(bsc))
        if not self.w3.isConnected():
            raise Exception('Web3 not connected!')
        self.contract = self._load_contract(abi_name='prediction', address=contract_address)
        self.address = address
        self.private_key = private_key
        self.nonce = self.w3.eth.getTransactionCount(self.address)
        if gas_optimizer:
            self.logger.info(f'Optimizing Gas Price...')
            self.w3.middleware_onion.inject(middleware.geth_poa_middleware, layer=0)
            strategy = construct_time_based_gas_price_strategy(self.max_wait_seconds, self.gas_sample_size)
            self.w3.eth.set_gas_price_strategy(strategy)

            # self.w3.middleware_onion.add(middleware.time_based_cache_middleware)
            # self.w3.middleware_onion.add(middleware.latest_block_based_cache_middleware)
            # self.w3.middleware_onion.add(middleware.simple_cache_middleware)
            # BUG: gas price should be decimal instead of float
            self.gas_price = float(self.w3.fromWei(self.w3.eth.generate_gas_price(), 'gwei'))
            self.logger.info(f'Optimzed Gas Price: {self.gas_price}')
        

    def _load_contract(self, abi_name, address):
        return self.w3.eth.contract(address=address, abi=self._load_abi(abi_name))

    @staticmethod
    def _load_abi(name: str) -> str:
        path = f'{os.path.dirname(os.path.abspath(__file__))}/assets/'
        with open(os.path.abspath(path + f'{name}.abi')) as f:
            abi: str = json.load(f)
        return abi

    def _get_tx_params(self, value=0):
        """Get generic transaction parameters."""
        return {
            "from": self.address,
            "value": value,
            "gas": self.gas,
            "gasPrice": self.w3.toWei(self.gas_price, 'gwei'),
            "nonce": max(self.nonce, self.w3.eth.getTransactionCount(self.address))
        }

    def _build_and_send_tx(self, function, tx_params=None):
        """Build and send a transaction."""
        if tx_params is None:
            tx_params = self._get_tx_params()
        tx = function.buildTransaction(tx_params)
        signed_txn = self.w3.eth.account.sign_transaction(tx, private_key=self.private_key)
        try:
            return self.w3.eth.sendRawTransaction(signed_txn.rawTransaction)
        finally:
            self.logger.debug(f"nonce: {tx_params['nonce']}")
            self.nonce = tx_params["nonce"] + 1

    def place_bet(self, bet_size, direction):
        if bet_size > self.min_bet_size:
            return self._build_and_send_tx(
                self.contract.get_function_by_name('bet'+direction)(),
                self._get_tx_params() | {
                    'value': self.w3.toWei(bet_size, 'ether')
                }
            )

    def claim_rewards(self, epoch, gas=120000, gas_price=5):
        if self.contract.caller.claimable(epoch, self.address):
            return self._build_and_send_tx(
                self.contract.functions.claim(epoch),
                self._get_tx_params() | {
                    'gas': gas,
                    'gasPrice': self.w3.toWei(gas_price, 'gwei')
                }
            )

    def compute_kelly(self, bull_odd, bear_odd):
        bull_kelly = (self.bull_win_rate*bull_odd-1)/(bull_odd-1)
        bear_kelly = ((1-self.bull_win_rate)*bear_odd-1)/(bear_odd-1)
        return bull_kelly, bear_kelly

    def start(self):
        
        prev_epoch = 0

        while True:

            curr_epoch = self.contract.caller.currentEpoch()
            bet_on = False if curr_epoch != prev_epoch else bet_on
            balance = self.balance_override if self.balance_override > 0 else float(
                self.w3.fromWei(self.w3.eth.get_balance(self.address), 'ether')) - self.gas_fee_reserve

            if balance < self.min_balance_size:
                sys.exit(f'Balance should not be less than {self.min_balance_size}')

            rounds = self.contract.caller.rounds(curr_epoch)
            blocks_away = rounds[2]-self.w3.eth.block_number
            bull_amount, bear_amount, reward_amount = rounds[7], rounds[8], rounds[10]
            prize_pool = float(self.w3.fromWei(reward_amount, 'ether'))
            gas_fee = self.gas*self.gas_price/2
            
            if blocks_away > 50 and prev_epoch > 0:
                tx_hash = self.claim_rewards(prev_epoch-1)
                if tx_hash and (receipt := self.w3.eth.wait_for_transaction_receipt(tx_hash)):
                    self.logger.info(f"Claim status: {receipt['status']}")

            if bull_amount > 0 and bear_amount > 0:
                bull_odd = (reward_amount-gas_fee)/bull_amount
                bear_odd = (reward_amount-gas_fee)/bear_amount
                
                bull_kelly, bear_kelly = self.compute_kelly(bull_odd, bear_odd)
                
                self.logger.info(f'Round: {curr_epoch} | Blocks Away: {blocks_away} | Bull Odds: {bull_odd:.3f} | Bull Kelly: {bull_kelly:.0%} | Bear Odds: {bear_odd:.3f} | Bear Kelly: {bear_kelly:.0%} | Prize Pool: {prize_pool:.3f} | Balance: {balance:.3f}')

                if prize_pool < self.min_prize_pool or not (1 < blocks_away <= self.execution_block):                    
                    time.sleep(self.polling_seconds)
                    continue
                
                max_kelly = max(bull_kelly, bear_kelly)
                if not bet_on and max_kelly >= self.min_kelly:
                    direction = 'Bull' if bull_kelly > bear_kelly else 'Bear'
                    bet_size = balance*max_kelly
                    try:
                        tx_hash = self.place_bet(bet_size, direction)
                        if tx_hash and (receipt := self.w3.eth.wait_for_transaction_receipt(tx_hash)):
                            self.logger.info(f"Place status: {receipt['status']}")
                            if receipt['status'] == 1:
                                bet_on = True
                                self.logger.info(f'A {bet_size:.2f} {direction} BNB Bet is placed!')
                            elif receipt['status'] == 0:
                                self.logger.info(f'A {bet_size:.2f} {direction} BNB Bet has not been placed!')
                                continue
                    except Exception as e:
                        self.logger.info(e)

            prev_epoch = curr_epoch


if __name__ == '__main__':
    
    logging.basicConfig(level = logging.INFO, format=('[%(levelname)s] %(message)s'))
    
    prediction = Prediction(address, private_key, gas_optimizer=False)
    prediction.start()

    logging.shutdown()
