"""
Credit Risk Models

Comprehensive credit risk assessment including:
- Counterparty risk assessment and monitoring
- Default probability estimation
- Credit exposure calculations
- Credit rating models
- Expected loss calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
import asyncio
from scipy import stats, optimize
from enum import Enum

from database.models.trading import Position, Trade
from database.models.user import User

logger = logging.getLogger(__name__)


class CreditRating(Enum):
    """Credit rating levels."""
    AAA = "AAA"
    AA = "AA+"
    AA_MINUS = "AA-"
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    BBB_PLUS = "BBB+"
    BBB = "BBB"
    BBB_MINUS = "BBB-"
    BB_PLUS = "BB+"
    BB = "BB"
    BB_MINUS = "BB-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    CCC_PLUS = "CCC+"
    CCC = "CCC"
    CCC_MINUS = "CCC-"
    CC = "CC"
    C = "C"
    D = "D"


@dataclass
class CreditExposure:
    """Credit exposure details."""
    counterparty_id: str
    exposure_amount: float
    exposure_type: str
    maturity_date: Optional[datetime]
    collateral_amount: float
    netting_set: str
    credit_rating: Optional[CreditRating]
    default_probability: float
    expected_loss: float
    unexpected_loss: float


@dataclass
class DefaultProbability:
    """Default probability result."""
    counterparty_id: str
    one_year_pd: float
    five_year_pd: float
    cumulative_pd: float
    hazard_rate: float
    survival_probability: float
    rating_migration_probabilities: Dict[str, float]
    methodology: str
    last_updated: datetime


class CounterpartyRisk:
    """
    Counterparty risk assessment and monitoring.
    
    Evaluates and monitors credit risk from counterparty exposures
    including banks, brokers, and trading partners.
    """
    
    def __init__(self, user_id: int):
        """
        Initialize counterparty risk assessment.
        
        Args:
            user_id: User identifier
        """
        self.user_id = user_id
        self.counterparties = {}
        self.exposures = {}
        
    async def assess_counterparty_risk(self, counterparty_id: str, 
                                     exposure_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess credit risk for specific counterparty.
        
        Args:
            counterparty_id: Counterparty identifier
            exposure_data: Counterparty exposure information
            
        Returns:
            Comprehensive counterparty risk assessment
        """
        try:
            # Calculate credit exposure
            credit_exposure = await self._calculate_credit_exposure(counterparty_id, exposure_data)
            
            # Estimate default probability
            default_prob = await self._estimate_default_probability(counterparty_id, exposure_data)
            
            # Calculate expected and unexpected loss
            loss_metrics = self._calculate_loss_metrics(credit_exposure, default_prob)
            
            # Assess concentration risk
            concentration_risk = await self._assess_concentration_risk(counterparty_id)
            
            # Generate risk rating
            risk_rating = self._generate_risk_rating(default_prob, concentration_risk)
            
            # Calculate capital requirement
            capital_req = self._calculate_capital_requirement(credit_exposure, default_prob)
            
            return {
                'counterparty_id': counterparty_id,
                'credit_exposure': credit_exposure,
                'default_probability': default_prob,
                'loss_metrics': loss_metrics,
                'concentration_risk': concentration_risk,
                'risk_rating': risk_rating,
                'capital_requirement': capital_req,
                'risk_score': self._calculate_overall_risk_score(loss_metrics, concentration_risk),
                'recommendations': self._generate_counterparty_recommendations(credit_exposure, risk_rating)
            }
            
        except Exception as e:
            logger.error(f"Counterparty risk assessment error: {str(e)}")
            return {'error': str(e), 'counterparty_id': counterparty_id}
    
    async def _calculate_credit_exposure(self, counterparty_id: str, 
                                       exposure_data: Dict[str, Any]) -> CreditExposure:
        """Calculate credit exposure to counterparty."""
        try:
            # Extract exposure components
            exposure_type = exposure_data.get('exposure_type', 'unsecured')
            current_exposure = exposure_data.get('current_exposure', 0.0)
            potential_future_exposure = exposure_data.get('potential_future_exposure', 0.0)
            collateral_amount = exposure_data.get('collateral_amount', 0.0)
            
            # Calculate total exposure
            gross_exposure = current_exposure + potential_future_exposure
            net_exposure = max(0, gross_exposure - collateral_amount)
            
            # Determine maturity
            maturity_date = exposure_data.get('maturity_date')
            if maturity_date and isinstance(maturity_date, str):
                maturity_date = datetime.strptime(maturity_date, '%Y-%m-%d')
            
            # Get counterparty rating
            rating = self._map_rating_to_enum(exposure_data.get('credit_rating'))
            
            # Calculate default probability
            default_prob = self._estimate_pd_from_rating(rating)
            
            # Calculate loss metrics
            expected_loss = net_exposure * default_prob
            unexpected_loss = self._calculate_unexpected_loss(net_exposure, default_prob)
            
            return CreditExposure(
                counterparty_id=counterparty_id,
                exposure_amount=net_exposure,
                exposure_type=exposure_type,
                maturity_date=maturity_date,
                collateral_amount=collateral_amount,
                netting_set=exposure_data.get('netting_set', 'default'),
                credit_rating=rating,
                default_probability=default_prob,
                expected_loss=expected_loss,
                unexpected_loss=unexpected_loss
            )
            
        except Exception as e:
            logger.error(f"Credit exposure calculation error: {str(e)}")
            return CreditExposure(
                counterparty_id=counterparty_id,
                exposure_amount=0.0,
                exposure_type='unsecured',
                maturity_date=None,
                collateral_amount=0.0,
                netting_set='default',
                credit_rating=None,
                default_probability=0.05,
                expected_loss=0.0,
                unexpected_loss=0.0
            )
    
    async def _estimate_default_probability(self, counterparty_id: str, 
                                          exposure_data: Dict[str, Any]) -> DefaultProbability:
        """Estimate default probability for counterparty."""
        try:
            # Get counterparty fundamentals
            rating = exposure_data.get('credit_rating')
            financial_metrics = exposure_data.get('financial_metrics', {})
            
            # Calculate PD based on rating
            rating_pd = self._estimate_pd_from_rating(self._map_rating_to_enum(rating))
            
            # Adjust PD based on financial metrics
            adjusted_pd = self._adjust_pd_for_financials(rating_pd, financial_metrics)
            
            # Calculate cumulative PD over time horizon
            time_horizon = exposure_data.get('time_horizon_years', 1.0)
            cumulative_pd = self._calculate_cumulative_pd(adjusted_pd, time_horizon)
            
            # Calculate hazard rate
            hazard_rate = self._calculate_hazard_rate(adjusted_pd, time_horizon)
            
            # Generate rating migration matrix
            migration_probs = self._generate_rating_migration_matrix(rating)
            
            return DefaultProbability(
                counterparty_id=counterparty_id,
                one_year_pd=adjusted_pd,
                five_year_pd=self._calculate_cumulative_pd(adjusted_pd, 5.0),
                cumulative_pd=cumulative_pd,
                hazard_rate=hazard_rate,
                survival_probability=1.0 - cumulative_pd,
                rating_migration_probabilities=migration_probs,
                methodology='rating_based_with_financial_adjustment',
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Default probability estimation error: {str(e)}")
            return DefaultProbability(
                counterparty_id=counterparty_id,
                one_year_pd=0.05,
                five_year_pd=0.20,
                cumulative_pd=0.05,
                hazard_rate=0.051,
                survival_probability=0.95,
                rating_migration_probabilities={},
                methodology='error_fallback',
                last_updated=datetime.now()
            )
    
    def _map_rating_to_enum(self, rating: str) -> Optional[CreditRating]:
        """Map rating string to CreditRating enum."""
        try:
            if not rating:
                return None
            
            rating_upper = rating.upper().strip()
            for credit_rating in CreditRating:
                if credit_rating.value == rating_upper:
                    return credit_rating
            
            return None
            
        except Exception:
            return None
    
    def _estimate_pd_from_rating(self, rating: Optional[CreditRating]) -> float:
        """Estimate default probability from credit rating."""
        if not rating:
            return 0.05  # Default 5% PD for unrated
        
        # PD mapping based on rating (approximate historical averages)
        pd_mapping = {
            CreditRating.AAA: 0.0001,
            CreditRating.AA: 0.0003,
            CreditRating.AA_MINUS: 0.0005,
            CreditRating.A_PLUS: 0.0010,
            CreditRating.A: 0.0020,
            CreditRating.A_MINUS: 0.0030,
            CreditRating.BBB_PLUS: 0.0050,
            CreditRating.BBB: 0.0080,
            CreditRating.BBB_MINUS: 0.0120,
            CreditRating.BB_PLUS: 0.0200,
            CreditRating.BB: 0.0350,
            CreditRating.BB_MINUS: 0.0600,
            CreditRating.B_PLUS: 0.1000,
            CreditRating.B: 0.1500,
            CreditRating.B_MINUS: 0.2000,
            CreditRating.CCC_PLUS: 0.3000,
            CreditRating.CCC: 0.4000,
            CreditRating.CCC_MINUS: 0.5000,
            CreditRating.CC: 0.6000,
            CreditRating.C: 0.7000,
            CreditRating.D: 1.0000
        }
        
        return pd_mapping.get(rating, 0.05)
    
    def _adjust_pd_for_financials(self, base_pd: float, financial_metrics: Dict[str, float]) -> float:
        """Adjust PD based on financial metrics."""
        try:
            if not financial_metrics:
                return base_pd
            
            adjustment_factor = 1.0
            
            # Leverage adjustment
            leverage = financial_metrics.get('debt_to_equity', 1.0)
            if leverage > 2.0:
                adjustment_factor *= 1.5
            elif leverage > 1.0:
                adjustment_factor *= 1.2
            
            # Interest coverage adjustment
            interest_coverage = financial_metrics.get('interest_coverage_ratio', 5.0)
            if interest_coverage < 2.0:
                adjustment_factor *= 1.8
            elif interest_coverage < 3.0:
                adjustment_factor *= 1.4
            
            # Current ratio adjustment
            current_ratio = financial_metrics.get('current_ratio', 2.0)
            if current_ratio < 1.0:
                adjustment_factor *= 1.6
            elif current_ratio < 1.5:
                adjustment_factor *= 1.3
            
            # ROE adjustment
            roe = financial_metrics.get('roe', 0.15)
            if roe < 0.05:
                adjustment_factor *= 1.4
            elif roe < 0.10:
                adjustment_factor *= 1.2
            
            return min(1.0, base_pd * adjustment_factor)
            
        except Exception as e:
            logger.error(f"PD adjustment error: {str(e)}")
            return base_pd
    
    def _calculate_cumulative_pd(self, annual_pd: float, years: float) -> float:
        """Calculate cumulative default probability over time period."""
        try:
            # Assume constant hazard rate
            hazard_rate = -np.log(1 - annual_pd)
            cumulative_survival = np.exp(-hazard_rate * years)
            cumulative_pd = 1 - cumulative_survival
            
            return min(1.0, cumulative_pd)
            
        except Exception:
            return min(1.0, annual_pd * years)
    
    def _calculate_hazard_rate(self, annual_pd: float, years: float) -> float:
        """Calculate hazard rate from default probability."""
        try:
            return -np.log(1 - annual_pd) / years if years > 0 else 0.0
        except Exception:
            return annual_pd
    
    def _generate_rating_migration_matrix(self, current_rating: str) -> Dict[str, float]:
        """Generate rating migration probability matrix."""
        try:
            # Simplified migration matrix (actual matrices are more complex)
            migration_probs = {rating.value: 0.0 for rating in CreditRating}
            
            if current_rating:
                # Set high probability for same rating
                migration_probs[current_rating] = 0.85
                
                # Set some probability for adjacent ratings
                rating_index = list(CreditRating).index(CreditRating[current_rating])
                
                # Adjacent rating probabilities
                if rating_index > 0:
                    prev_rating = list(CreditRating)[rating_index - 1]
                    migration_probs[prev_rating.value] = 0.08
                
                if rating_index < len(list(CreditRating)) - 1:
                    next_rating = list(CreditRating)[rating_index + 1]
                    migration_probs[next_rating.value] = 0.05
                
                # Default probability
                migration_probs['D'] = 0.02
            else:
                # For unrated, use default matrix
                migration_probs['BBB'] = 0.80
                migration_probs['BB'] = 0.10
                migration_probs['B'] = 0.05
                migration_probs['D'] = 0.05
            
            return migration_probs
            
        except Exception as e:
            logger.error(f"Rating migration matrix generation error: {str(e)}")
            return {}
    
    def _calculate_loss_metrics(self, exposure: CreditExposure, 
                              default_prob: DefaultProbability) -> Dict[str, float]:
        """Calculate expected and unexpected loss metrics."""
        try:
            exposure_amount = exposure.exposure_amount
            pd = default_prob.one_year_pd
            
            # Expected Loss (EL)
            expected_loss = exposure_amount * pd
            
            # Loss Given Default (LGD) - simplified at 60%
            lgd = 0.60
            expected_loss *= lgd
            
            # Unexpected Loss (UL) using binomial approximation
            variance = exposure_amount * pd * (1 - pd) * (lgd ** 2)
            unexpected_loss = np.sqrt(variance) * 2.33  # 99% confidence
            
            return {
                'expected_loss': expected_loss,
                'unexpected_loss': unexpected_loss,
                'loss_given_default': lgd,
                'exposure_at_default': exposure_amount * lgd,
                'economic_capital': unexpected_loss * 1.5  # Additional buffer
            }
            
        except Exception as e:
            logger.error(f"Loss metrics calculation error: {str(e)}")
            return {}
    
    async def _assess_concentration_risk(self, counterparty_id: str) -> Dict[str, float]:
        """Assess concentration risk to counterparty."""
        try:
            # Get total portfolio exposure
            total_portfolio_exposure = sum(
                exp.exposure_amount for exp in self.exposures.values()
            )
            
            # Get counterparty exposure
            counterparty_exposure = self.exposures.get(counterparty_id)
            if not counterparty_exposure:
                return {'concentration_ratio': 0.0, 'concentration_risk_level': 'low'}
            
            # Calculate concentration ratio
            concentration_ratio = (
                counterparty_exposure.exposure_amount / total_portfolio_exposure
                if total_portfolio_exposure > 0 else 0
            )
            
            # Assess concentration risk level
            if concentration_ratio > 0.25:
                risk_level = 'extreme'
            elif concentration_ratio > 0.15:
                risk_level = 'high'
            elif concentration_ratio > 0.10:
                risk_level = 'medium'
            else:
                risk_level = 'low'
            
            return {
                'concentration_ratio': concentration_ratio,
                'concentration_risk_level': risk_level,
                'total_portfolio_exposure': total_portfolio_exposure,
                'counterparty_exposure': counterparty_exposure.exposure_amount
            }
            
        except Exception as e:
            logger.error(f"Concentration risk assessment error: {str(e)}")
            return {'concentration_ratio': 0.0, 'concentration_risk_level': 'unknown'}
    
    def _generate_risk_rating(self, default_prob: DefaultProbability, 
                            concentration_risk: Dict[str, float]) -> Dict[str, Any]:
        """Generate risk rating for counterparty."""
        try:
            # Base rating from default probability
            base_rating = self._rating_from_pd(default_prob.one_year_pd)
            
            # Adjust for concentration risk
            concentration_adjustment = 0
            risk_level = concentration_risk.get('concentration_risk_level', 'low')
            
            if risk_level == 'extreme':
                concentration_adjustment = 2
            elif risk_level == 'high':
                concentration_adjustment = 1
            
            # Apply adjustment
            adjusted_rating = self._adjust_rating(base_rating, concentration_adjustment)
            
            return {
                'base_rating': base_rating,
                'adjusted_rating': adjusted_rating,
                'rating_methodology': 'pd_based_with_concentration_adjustment',
                'confidence_level': 'high' if default_prob.one_year_pd < 0.1 else 'medium',
                'last_review_date': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Risk rating generation error: {str(e)}")
            return {'base_rating': 'BBB', 'adjusted_rating': 'BBB', 'rating_methodology': 'error_fallback'}
    
    def _rating_from_pd(self, pd: float) -> str:
        """Map default probability to credit rating."""
        if pd <= 0.0002:
            return 'AAA'
        elif pd <= 0.0005:
            return 'AA'
        elif pd <= 0.001:
            return 'AA-'
        elif pd <= 0.002:
            return 'A+'
        elif pd <= 0.004:
            return 'A'
        elif pd <= 0.006:
            return 'A-'
        elif pd <= 0.008:
            return 'BBB+'
        elif pd <= 0.015:
            return 'BBB'
        elif pd <= 0.025:
            return 'BBB-'
        elif pd <= 0.040:
            return 'BB+'
        elif pd <= 0.070:
            return 'BB'
        elif pd <= 0.120:
            return 'BB-'
        elif pd <= 0.180:
            return 'B+'
        elif pd <= 0.250:
            return 'B'
        elif pd <= 0.350:
            return 'B-'
        elif pd <= 0.500:
            return 'CCC'
        else:
            return 'D'
    
    def _adjust_rating(self, rating: str, adjustment: int) -> str:
        """Adjust rating based on risk factors."""
        try:
            ratings_order = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 
                           'BBB', 'BBB-', 'BB+', 'BB', 'BB-', 'B+', 'B', 'B-',
                           'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
            
            if rating not in ratings_order:
                return rating
            
            current_index = ratings_order.index(rating)
            adjusted_index = max(0, min(len(ratings_order) - 1, current_index + adjustment))
            
            return ratings_order[adjusted_index]
            
        except Exception:
            return rating
    
    def _calculate_capital_requirement(self, exposure: CreditExposure, 
                                     default_prob: DefaultProbability) -> Dict[str, float]:
        """Calculate regulatory capital requirement."""
        try:
            # Simplified Basel II capital calculation
            exposure_amount = exposure.exposure_amount
            pd = default_prob.one_year_pd
            
            # Capital requirement factors
            if pd <= 0.003:
                capital_factor = 1.5  # Low risk
            elif pd <= 0.01:
                capital_factor = 3.0  # Medium risk
            elif pd <= 0.05:
                capital_factor = 6.0  # Higher risk
            else:
                capital_factor = 12.0  # High risk
            
            # Calculate capital requirement
            capital_requirement = exposure_amount * pd * capital_factor
            
            return {
                'regulatory_capital': capital_requirement,
                'capital_factor': capital_factor,
                'risk_weighted_assets': exposure_amount * capital_factor,
                'capital_ratio': capital_requirement / exposure_amount if exposure_amount > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Capital requirement calculation error: {str(e)}")
            return {}
    
    def _calculate_overall_risk_score(self, loss_metrics: Dict[str, float], 
                                    concentration_risk: Dict[str, float]) -> float:
        """Calculate overall risk score (0-100)."""
        try:
            # Base score from unexpected loss
            ul_score = min(50, loss_metrics.get('unexpected_loss', 0) / 1000)  # Scale to 50 points
            
            # Concentration score
            conc_ratio = concentration_risk.get('concentration_ratio', 0)
            conc_score = min(50, conc_ratio * 200)  # Scale to 50 points
            
            # Total risk score
            total_score = ul_score + conc_score
            
            return min(100, total_score)
            
        except Exception:
            return 50.0  # Default medium risk score
    
    def _generate_counterparty_recommendations(self, exposure: CreditExposure, 
                                             risk_rating: Dict[str, Any]) -> List[str]:
        """Generate recommendations for counterparty risk management."""
        try:
            recommendations = []
            
            adjusted_rating = risk_rating.get('adjusted_rating', 'BBB')
            exposure_amount = exposure.exposure_amount
            
            # Rating-based recommendations
            if adjusted_rating in ['BB', 'BB-', 'B+', 'B', 'B-', 'CCC', 'D']:
                recommendations.append("Consider reducing exposure to lower-rated counterparty")
                recommendations.append("Implement enhanced monitoring for this counterparty")
            
            if adjusted_rating == 'D':
                recommendations.append("URGENT: Avoid new transactions with defaulted counterparty")
                recommendations.append("Initiate recovery procedures immediately")
            
            # Exposure-based recommendations
            if exposure_amount > 1000000:  # $1M+
                recommendations.append("Large exposure - consider additional collateral")
                recommendations.append("Review counterparty limits")
            
            # Concentration recommendations
            if exposure_amount > 500000:  # $500k+
                recommendations.append("Monitor concentration limits closely")
            
            # Maturity recommendations
            if exposure.maturity_date:
                days_to_maturity = (exposure.maturity_date - datetime.now()).days
                if days_to_maturity < 30:
                    recommendations.append("Counterparty exposure maturing soon - plan renewal or exit")
                elif days_to_maturity < 90:
                    recommendations.append("Review counterparty status before maturity")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Counterparty recommendations generation error: {str(e)}")
            return ["Review counterparty risk profile"]


class DefaultProbability:
    """
    Advanced default probability estimation models.
    
    Implements multiple methodologies for estimating default
    probability including statistical models and market-based approaches.
    """
    
    def __init__(self, user_id: int):
        """Initialize default probability calculator."""
        self.user_id = user_id
        
    async def estimate_structural_pd(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate default probability using structural model (Merton-type).
        
        Args:
            financial_data: Financial statement data
            
        Returns:
            Structural model PD estimate
        """
        try:
            # Extract financial variables
            asset_value = financial_data.get('total_assets', 1000000)
            debt_value = financial_data.get('total_debt', 500000)
            asset_volatility = financial_data.get('asset_volatility', 0.20)
            risk_free_rate = financial_data.get('risk_free_rate', 0.03)
            
            # Calculate asset value and distance to default
            equity_value = financial_data.get('market_cap', asset_value - debt_value)
            debt_maturity = financial_data.get('debt_maturity', 1.0)
            
            # Merton model calculations
            if asset_value <= debt_value:
                return {'pd': 0.5, 'distance_to_default': 0.0, 'method': 'structural_merton'}
            
            # Approximate asset volatility from equity volatility
            debt_to_equity = debt_value / equity_value if equity_value > 0 else 1.0
            asset_vol_approx = asset_volatility / (1 + debt_to_equity)
            
            # Distance to default
            if asset_vol_approx > 0:
                distance_to_default = (
                    np.log(asset_value / debt_value) + 
                    (risk_free_rate - 0.5 * asset_vol_approx ** 2) * debt_maturity
                ) / (asset_vol_approx * np.sqrt(debt_maturity))
            else:
                distance_to_default = 0.0
            
            # Convert to default probability
            default_probability = stats.norm.cdf(-distance_to_default)
            
            return {
                'pd': default_probability,
                'distance_to_default': distance_to_default,
                'asset_value': asset_value,
                'debt_value': debt_value,
                'asset_volatility': asset_vol_approx,
                'method': 'structural_merton',
                'confidence': 'high' if equity_value > debt_value else 'low'
            }
            
        except Exception as e:
            logger.error(f"Structural PD estimation error: {str(e)}")
            return {'pd': 0.05, 'distance_to_default': 0.0, 'method': 'error_fallback'}
    
    async def estimate_reduced_form_pd(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate default probability using reduced-form (intensity) model.
        
        Args:
            market_data: Market-based data (CDS spreads, bond yields, etc.)
            
        Returns:
            Reduced-form model PD estimate
        """
        try:
            cds_spread = market_data.get('cds_spread', 0.02)
            bond_yield = market_data.get('bond_yield', 0.05)
            risk_free_rate = market_data.get('risk_free_rate', 0.03)
            maturity = market_data.get('maturity_years', 1.0)
            
            # Convert CDS spread to hazard rate
            if cds_spread > 0:
                hazard_rate = cds_spread
            else:
                # Fallback: use bond yield spread
                bond_spread = bond_yield - risk_free_rate
                hazard_rate = max(0.001, bond_spread)
            
            # Calculate cumulative default probability
            cumulative_pd = 1 - np.exp(-hazard_rate * maturity)
            
            # Calculate conditional default probability for 1 year
            annual_pd = 1 - np.exp(-hazard_rate)
            
            return {
                'pd': annual_pd,
                'cumulative_pd': cumulative_pd,
                'hazard_rate': hazard_rate,
                'cds_spread': cds_spread,
                'bond_spread': bond_yield - risk_free_rate if 'bond_yield' in market_data else None,
                'method': 'reduced_form_intensity',
                'confidence': 'high' if cds_spread > 0 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Reduced-form PD estimation error: {str(e)}")
            return {'pd': 0.05, 'cumulative_pd': 0.05, 'hazard_rate': 0.051, 'method': 'error_fallback'}
    
    async def estimate_machine_learning_pd(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate default probability using machine learning models.
        
        Args:
            features: Feature vector for ML model
            
        Returns:
            ML-based PD estimate
        """
        try:
            # Simplified ML PD estimation (in practice, would use trained models)
            
            # Extract key features
            leverage = features.get('debt_to_equity', 1.0)
            roa = features.get('return_on_assets', 0.05)
            current_ratio = features.get('current_ratio', 2.0)
            profit_margin = features.get('profit_margin', 0.10)
            interest_coverage = features.get('interest_coverage', 5.0)
            
            # Simple logistic regression approximation
            z_score = (
                -2.0 +  # Intercept
                1.5 * max(0, leverage - 1.0) +  # Leverage effect
                -8.0 * max(0, roa) +  # Profitability effect
                -0.5 * max(0, current_ratio - 1.0) +  # Liquidity effect
                -3.0 * max(0, profit_margin) +  # Margin effect
                -0.1 * max(0, interest_coverage - 1.0)  # Coverage effect
            )
            
            # Convert to probability using sigmoid
            pd = 1 / (1 + np.exp(-z_score))
            
            # Confidence based on feature completeness
            feature_count = sum(1 for v in features.values() if v is not None)
            total_features = 6
            confidence = min(1.0, feature_count / total_features)
            
            return {
                'pd': pd,
                'z_score': z_score,
                'confidence': confidence,
                'feature_importance': {
                    'leverage': 1.5,
                    'roa': 8.0,
                    'current_ratio': 0.5,
                    'profit_margin': 3.0,
                    'interest_coverage': 0.1
                },
                'method': 'machine_learning_logistic',
                'model_type': 'logistic_regression_simplified'
            }
            
        except Exception as e:
            logger.error(f"ML PD estimation error: {str(e)}")
            return {'pd': 0.10, 'confidence': 0.0, 'method': 'error_fallback'}


class CreditRiskFactory:
    """Factory class for creating credit risk models."""
    
    @staticmethod
    def create_model(model_type: str, user_id: int, **kwargs) -> Any:
        """
        Create credit risk model instance.
        
        Args:
            model_type: Type of credit risk model
            user_id: User identifier
            **kwargs: Model-specific parameters
            
        Returns:
            Credit risk model instance
        """
        model_types = {
            'counterparty': CounterpartyRisk,
            'default_probability': DefaultProbability
        }
        
        if model_type.lower() not in model_types:
            raise ValueError(f"Unsupported credit risk model type: {model_type}")
        
        return model_types[model_type.lower()](user_id, **kwargs)
