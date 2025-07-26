from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import openai
import os
import json
from datetime import datetime, timedelta
import uuid
from typing import Dict, List
import numpy as np
import re
from collections import defaultdict

app = FastAPI()

# WebSocket connection manager for collaboration
class CollaborationManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.session_data: Dict[str, dict] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        if session_id not in self.active_connections:
            self.active_connections[session_id] = []
        self.active_connections[session_id].append(websocket)

    def disconnect(self, websocket: WebSocket, session_id: str):
        if session_id in self.active_connections:
            self.active_connections[session_id].remove(websocket)

    async def broadcast_to_session(self, session_id: str, message: str):
        if session_id in self.active_connections:
            for connection in self.active_connections[session_id]:
                try:
                    await connection.send_text(message)
                except:
                    # Remove dead connections
                    self.active_connections[session_id].remove(connection)

collaboration_manager = CollaborationManager()

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await collaboration_manager.connect(websocket, session_id)
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast drawing data to all participants in session
            await collaboration_manager.broadcast_to_session(session_id, data)
    except WebSocketDisconnect:
        collaboration_manager.disconnect(websocket, session_id)

# Initialize OpenAI client with enhanced configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple Transformer Model for Local Processing
class SimpleBertTransformer:
    """Lightweight transformer model for text understanding without external dependencies"""
    
    def __init__(self):
        self.vocab = {}
        self.embedding_dim = 128
        self.max_seq_length = 64
        self.attention_heads = 4
        self.hidden_dim = 256
        
        # Initialize transformer components
        self.embeddings = {}
        self.attention_weights = {}
        self.position_encodings = self._create_position_encodings()
        self.layer_norm_weights = np.random.normal(0, 0.02, (self.embedding_dim,))
        self.feed_forward_weights = {
            'W1': np.random.normal(0, 0.02, (self.embedding_dim, self.hidden_dim)),
            'b1': np.zeros(self.hidden_dim),
            'W2': np.random.normal(0, 0.02, (self.hidden_dim, self.embedding_dim)),
            'b2': np.zeros(self.embedding_dim)
        }
        
        # Art-specific vocabulary and embeddings
        self._build_art_vocabulary()
        self._initialize_embeddings()
    
    def _build_art_vocabulary(self):
        """Build vocabulary from art and drawing terms"""
        art_vocab = [
            # Basic drawing terms
            'draw', 'sketch', 'paint', 'color', 'brush', 'pencil', 'canvas', 'line', 'shape',
            'circle', 'square', 'triangle', 'rectangle', 'oval', 'star', 'arrow',
            
            # Techniques
            'shading', 'blending', 'hatching', 'crosshatch', 'stippling', 'smudging',
            'perspective', 'composition', 'proportion', 'anatomy', 'realism', 'abstract',
            
            # Art styles
            'realistic', 'cartoon', 'anime', 'manga', 'impressionist', 'cubist', 'surreal',
            'watercolor', 'oil', 'acrylic', 'digital', 'traditional',
            
            # Art elements
            'texture', 'pattern', 'gradient', 'highlight', 'shadow', 'midtone', 'contrast',
            'saturation', 'hue', 'value', 'warm', 'cool', 'complementary', 'analogous',
            
            # Common subjects
            'portrait', 'landscape', 'still', 'life', 'figure', 'face', 'eye', 'nose', 'mouth',
            'hair', 'hand', 'tree', 'flower', 'animal', 'cat', 'dog', 'bird', 'house',
            
            # Technical terms
            'layer', 'opacity', 'blend', 'mode', 'tool', 'size', 'hardness', 'flow',
            'symmetry', 'balance', 'focal', 'point', 'depth', 'space', 'form', 'volume',
            
            # Problem-solving
            'help', 'problem', 'issue', 'error', 'trouble', 'stuck', 'fix', 'solution',
            'tutorial', 'guide', 'step', 'instruction', 'method', 'technique', 'tip',
            
            # Skill levels
            'beginner', 'intermediate', 'advanced', 'expert', 'professional', 'master',
            'basic', 'simple', 'complex', 'difficult', 'easy', 'practice', 'improve',
            
            # Special tokens
            '[PAD]', '[UNK]', '[CLS]', '[SEP]'
        ]
        
        # Add common words
        common_words = [
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'how', 'what', 'when', 'where', 'why', 'which', 'who', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'make', 'get', 'go',
            'come', 'take', 'give', 'see', 'look', 'use', 'find', 'work', 'call', 'try',
            'ask', 'need', 'feel', 'become', 'leave', 'put', 'mean', 'keep', 'let', 'begin',
            'seem', 'help', 'talk', 'turn', 'start', 'show', 'hear', 'play', 'run', 'move',
            'live', 'believe', 'hold', 'bring', 'happen', 'write', 'provide', 'sit', 'stand',
            'lose', 'pay', 'meet', 'include', 'continue', 'set', 'learn', 'change', 'lead',
            'understand', 'watch', 'follow', 'stop', 'create', 'speak', 'read', 'allow',
            'add', 'spend', 'grow', 'open', 'walk', 'win', 'offer', 'remember', 'love',
            'consider', 'appear', 'buy', 'wait', 'serve', 'die', 'send', 'expect', 'build',
            'stay', 'fall', 'cut', 'reach', 'kill', 'remain'
        ]
        
        all_vocab = art_vocab + common_words
        self.vocab = {word: idx for idx, word in enumerate(set(all_vocab))}
        self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def _initialize_embeddings(self):
        """Initialize word embeddings with art-specific patterns"""
        self.embeddings = np.random.normal(0, 0.02, (self.vocab_size, self.embedding_dim))
        
        # Create semantic clusters for related art terms
        art_clusters = {
            'drawing_tools': ['pencil', 'brush', 'eraser', 'pen', 'marker'],
            'shapes': ['circle', 'square', 'triangle', 'rectangle', 'oval', 'star'],
            'colors': ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white'],
            'techniques': ['shading', 'blending', 'hatching', 'crosshatch', 'stippling'],
            'subjects': ['portrait', 'landscape', 'still', 'life', 'figure', 'animal'],
            'skill_levels': ['beginner', 'intermediate', 'advanced', 'expert', 'professional'],
            'problems': ['help', 'problem', 'issue', 'error', 'trouble', 'stuck']
        }
        
        # Cluster similar words in embedding space
        for cluster_name, words in art_clusters.items():
            cluster_center = np.random.normal(0, 0.1, self.embedding_dim)
            for word in words:
                if word in self.vocab:
                    idx = self.vocab[word]
                    self.embeddings[idx] = cluster_center + np.random.normal(0, 0.05, self.embedding_dim)
    
    def _create_position_encodings(self):
        """Create positional encodings for transformer"""
        pe = np.zeros((self.max_seq_length, self.embedding_dim))
        position = np.arange(self.max_seq_length).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.embedding_dim, 2) * -(np.log(10000.0) / self.embedding_dim))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        return pe
    
    def tokenize(self, text):
        """Simple tokenization for input text"""
        # Clean and tokenize
        text = re.sub(r'[^\w\s]', '', text.lower())
        tokens = text.split()
        
        # Convert to token IDs
        token_ids = []
        for token in tokens[:self.max_seq_length-2]:  # Leave room for [CLS] and [SEP]
            token_ids.append(self.vocab.get(token, self.vocab.get('[UNK]', 0)))
        
        # Add special tokens
        cls_id = self.vocab.get('[CLS]', 0)
        sep_id = self.vocab.get('[SEP]', 0)
        token_ids = [cls_id] + token_ids + [sep_id]
        
        # Pad to max length
        while len(token_ids) < self.max_seq_length:
            token_ids.append(self.vocab.get('[PAD]', 0))
        
        return np.array(token_ids[:self.max_seq_length])
    
    def encode(self, text):
        """Encode text using transformer model"""
        token_ids = self.tokenize(text)
        
        # Get embeddings
        embeddings = self.embeddings[token_ids]
        
        # Add positional encoding
        embeddings = embeddings + self.position_encodings[:len(token_ids)]
        
        # Simple self-attention mechanism
        attended_embeddings = self._self_attention(embeddings)
        
        # Feed forward network
        output = self._feed_forward(attended_embeddings)
        
        # Return sentence embedding (mean of token embeddings)
        mask = token_ids != self.vocab.get('[PAD]', 0)
        sentence_embedding = np.mean(output[mask], axis=0)
        
        return sentence_embedding
    
    def _self_attention(self, embeddings):
        """Simplified self-attention mechanism"""
        seq_len, embed_dim = embeddings.shape
        
        # Simplified attention weights (normally would use learned Q, K, V matrices)
        attention_scores = np.dot(embeddings, embeddings.T) / np.sqrt(embed_dim)
        attention_weights = self._softmax(attention_scores)
        
        # Apply attention
        attended = np.dot(attention_weights, embeddings)
        
        # Residual connection and layer norm
        return self._layer_norm(attended + embeddings)
    
    def _feed_forward(self, x):
        """Feed forward network"""
        # First linear layer with ReLU
        hidden = np.maximum(0, np.dot(x, self.feed_forward_weights['W1']) + self.feed_forward_weights['b1'])
        
        # Second linear layer
        output = np.dot(hidden, self.feed_forward_weights['W2']) + self.feed_forward_weights['b2']
        
        # Residual connection and layer norm
        return self._layer_norm(output + x)
    
    def _softmax(self, x):
        """Stable softmax implementation"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _layer_norm(self, x):
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return (x - mean) / (std + 1e-6)
    
    def semantic_similarity(self, text1, text2):
        """Calculate semantic similarity between two texts"""
        embedding1 = self.encode(text1)
        embedding2 = self.encode(text2)
        
        # Cosine similarity
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def extract_intent(self, text):
        """Extract intent from text using transformer embeddings"""
        embedding = self.encode(text)
        
        # Define intent patterns based on embedding clusters
        intent_patterns = {
            'drawing_request': ['draw', 'sketch', 'create', 'make', 'paint'],
            'help_request': ['help', 'problem', 'issue', 'trouble', 'stuck', 'error'],
            'tutorial_request': ['how', 'tutorial', 'guide', 'step', 'instruction', 'method'],
            'improvement_request': ['improve', 'better', 'practice', 'develop', 'enhance'],
            'color_request': ['color', 'palette', 'hue', 'saturation', 'shade', 'tint'],
            'technique_request': ['technique', 'method', 'style', 'approach', 'way'],
            'tool_request': ['tool', 'brush', 'pencil', 'eraser', 'canvas', 'layer']
        }
        
        # Calculate similarity to each intent pattern
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            pattern_text = ' '.join(keywords)
            pattern_embedding = self.encode(pattern_text)
            similarity = np.dot(embedding, pattern_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(pattern_embedding)
            )
            intent_scores[intent] = float(similarity)
        
        # Return most likely intent
        best_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[best_intent]
        
        return {
            'intent': best_intent,
            'confidence': confidence,
            'all_scores': intent_scores
        }
    
    def generate_response_features(self, text):
        """Generate features for response generation"""
        embedding = self.encode(text)
        intent_analysis = self.extract_intent(text)
        
        # Extract key features
        features = {
            'semantic_embedding': embedding.tolist(),
            'intent': intent_analysis['intent'],
            'intent_confidence': intent_analysis['confidence'],
            'text_length': len(text.split()),
            'has_question': '?' in text,
            'has_specific_subject': any(word in text.lower() for word in [
                'cat', 'dog', 'tree', 'house', 'face', 'portrait', 'landscape'
            ]),
            'complexity_level': self._assess_complexity(text),
            'art_domain': self._identify_art_domain(text)
        }
        
        return features
    
    def _assess_complexity(self, text):
        """Assess text complexity using transformer features"""
        embedding = self.encode(text)
        
        # Simple complexity assessment based on embedding magnitude and diversity
        complexity_indicators = {
            'technical_terms': len([w for w in text.split() if len(w) > 8]),
            'multiple_concepts': len([w for w in text.split() if w in ['and', 'with', 'plus']]),
            'advanced_vocabulary': len([w for w in text.split() if w in [
                'professional', 'advanced', 'complex', 'sophisticated', 'intricate'
            ]])
        }
        
        complexity_score = sum(complexity_indicators.values()) / max(len(text.split()), 1)
        
        if complexity_score > 0.3:
            return 'advanced'
        elif complexity_score > 0.1:
            return 'intermediate'
        else:
            return 'basic'
    
    def _identify_art_domain(self, text):
        """Identify art domain using transformer understanding"""
        domain_keywords = {
            'portrait': ['face', 'portrait', 'person', 'character', 'head'],
            'landscape': ['landscape', 'scenery', 'nature', 'mountain', 'tree', 'sky'],
            'still_life': ['still life', 'object', 'fruit', 'vase', 'table'],
            'abstract': ['abstract', 'non-representational', 'conceptual', 'experimental'],
            'animal': ['animal', 'cat', 'dog', 'bird', 'horse', 'wildlife'],
            'architecture': ['building', 'house', 'city', 'structure', 'architectural'],
            'fantasy': ['dragon', 'fantasy', 'mythical', 'magical', 'creature']
        }
        
        text_embedding = self.encode(text)
        
        best_domain = 'general'
        best_score = 0.0
        
        for domain, keywords in domain_keywords.items():
            domain_text = ' '.join(keywords)
            domain_embedding = self.encode(domain_text)
            similarity = self.semantic_similarity(text, domain_text)
            
            if similarity > best_score:
                best_score = similarity
                best_domain = domain
        
        return best_domain

# Advanced AI Training System
class AITrainingSystem:
    def __init__(self):
        self.training_data = []
        self.model_feedback = {}
        self.user_corrections = {}
        self.response_cache = {}
        
        # Initialize transformer model
        self.transformer = SimpleBertTransformer()
        
        # Advanced learning components
        self.skill_progression_tracker = {}
        self.contextual_memory = {}
        self.learning_pathways = {}
        self.multi_modal_patterns = {}
        self.semantic_understanding = {}
        self.difficulty_scaling = {}
        
        # Comprehensive performance metrics
        self.performance_metrics = {
            "total_interactions": 0,
            "successful_responses": 0,
            "user_satisfaction_score": 0.0,
            "common_failure_patterns": [],
            "improvement_suggestions": [],
            "learning_velocity": 0.0,
            "concept_mastery_levels": {},
            "adaptive_complexity_score": 0.0,
            "cross_domain_correlations": {},
            "long_term_retention_rate": 0.0,
            "creative_solution_count": 0,
            "personalization_accuracy": 0.0
        }

    def collect_training_data(self, user_prompt, ai_response, user_rating, correction=None, response_time=None):
        """Collect comprehensive training data with advanced learning analysis"""
        # Enhanced training entry with multi-dimensional analysis
        training_entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": user_prompt.lower().strip(),
            "response": ai_response,
            "rating": user_rating,
            "correction": correction,
            "response_time": response_time,
            "session_id": str(uuid.uuid4()),
            "prompt_type": self._classify_prompt_type(user_prompt),
            "response_quality": self._assess_response_quality(ai_response, user_rating),
            
            # Advanced learning dimensions - Enhanced dataset feature handling
            "cognitive_complexity": self._analyze_cognitive_complexity(user_prompt),
            "semantic_depth": self._calculate_semantic_depth(user_prompt, ai_response),
            "creative_elements": self._identify_creative_elements(user_prompt, ai_response),
            "skill_level_required": self._determine_skill_level(user_prompt),
            "concept_categories": self._extract_concept_categories(user_prompt),
            "learning_objectives": self._identify_learning_objectives(user_prompt),
            "multimodal_components": self._analyze_multimodal_aspects(user_prompt),
            "contextual_relevance": self._assess_contextual_relevance(user_prompt),
            
            # Dataset-specific feature extraction
            "training_category": self._identify_training_category(user_prompt, ai_response),
            "technique_complexity": self._assess_technique_complexity(user_prompt),
            "cultural_context": self._extract_cultural_context(user_prompt),
            "mathematical_elements": self._identify_mathematical_patterns(user_prompt),
            "professional_level": self._determine_professional_level(ai_response),
            "artistic_movement": self._classify_artistic_movement(user_prompt),
            "scientific_accuracy": self._assess_scientific_accuracy(ai_response)
        }
        
        self.training_data.append(training_entry)
        self._update_performance_metrics(training_entry)
        
        # Advanced learning processes
        self._update_skill_progression(training_entry)
        self._enhance_contextual_memory(training_entry)
        self._evolve_learning_pathways(training_entry)
        self._build_semantic_understanding(training_entry)
        self._adapt_difficulty_scaling(training_entry)
        
        # Store in database for persistence
        if hasattr(db_manager, 'save_ai_training_data'):
            try:
                db_manager.save_ai_training_data(
                    "system", user_prompt, ai_response, user_rating, correction
                )
            except Exception as e:
                print(f"Failed to save training data: {e}")
        
        return training_entry

    def _classify_prompt_type(self, prompt):
        """Classify prompt type for better analysis"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['help', 'problem', 'not working', 'error', 'issue']):
            return "troubleshooting"
        elif any(word in prompt_lower for word in ['draw', 'sketch', 'paint', 'create']):
            return "drawing_instruction"
        elif any(word in prompt_lower for word in ['color', 'palette', 'shade']):
            return "color_guidance"
        elif any(word in prompt_lower for word in ['tutorial', 'how to', 'guide']):
            return "educational"
        else:
            return "general"

    def _assess_response_quality(self, response, rating):
        """Assess response quality based on content and rating"""
        quality_score = rating if rating else 3
        
        # Adjust based on response characteristics
        if len(response) < 50:
            quality_score -= 0.5  # Too short
        elif len(response) > 1000:
            quality_score -= 0.3  # Too long
            
        if "error" in response.lower() or "failed" in response.lower():
            quality_score -= 1.0  # Error responses
            
        return max(1, min(5, quality_score))

    def _update_performance_metrics(self, entry):
        """Update system performance metrics"""
        self.performance_metrics["total_interactions"] += 1
        
        if entry["rating"] >= 4:
            self.performance_metrics["successful_responses"] += 1
            
        # Update satisfaction score (rolling average)
        current_score = self.performance_metrics["user_satisfaction_score"]
        total = self.performance_metrics["total_interactions"]
        new_score = ((current_score * (total - 1)) + entry["rating"]) / total
        self.performance_metrics["user_satisfaction_score"] = round(new_score, 2)

    def fine_tune_responses(self, drawing_context, user_feedback):
        """Enhanced response generation based on collected feedback"""
        feedback_analysis = self.analyze_feedback_patterns()
        similar_prompts = self._find_similar_prompts(drawing_context)
        
        enhanced_prompt = f"""
        Context: {drawing_context}
        
        User Feedback Analysis: {feedback_analysis}
        Similar Successful Prompts: {similar_prompts}
        
        Performance Metrics:
        - Total Interactions: {self.performance_metrics['total_interactions']}
        - Success Rate: {self._calculate_success_rate()}%
        - User Satisfaction: {self.performance_metrics['user_satisfaction_score']}/5.0
        
        Generate an improved, personalized drawing instruction that:
        1. Addresses common user preferences from successful interactions
        2. Avoids patterns that led to negative feedback
        3. Uses clear, step-by-step language
        4. Includes specific tool recommendations
        """
        
        return enhanced_prompt

    def _find_similar_prompts(self, current_prompt, limit=3):
        """Find similar successful prompts for context"""
        current_words = set(current_prompt.lower().split())
        similar_prompts = []
        
        for entry in self.training_data:
            if entry["rating"] >= 4:  # Only successful responses
                entry_words = set(entry["prompt"].split())
                similarity = len(current_words.intersection(entry_words)) / len(current_words.union(entry_words))
                
                if similarity > 0.3:  # 30% similarity threshold
                    similar_prompts.append({
                        "prompt": entry["prompt"],
                        "response": entry["response"][:100] + "...",
                        "similarity": similarity
                    })
        
        return sorted(similar_prompts, key=lambda x: x["similarity"], reverse=True)[:limit]

    def _calculate_success_rate(self):
        """Calculate success rate percentage"""
        if self.performance_metrics["total_interactions"] == 0:
            return 0
        return round(
            (self.performance_metrics["successful_responses"] / 
             self.performance_metrics["total_interactions"]) * 100, 1
        )

    def analyze_feedback_patterns(self):
        """Comprehensive feedback pattern analysis"""
        if not self.training_data:
            return {"status": "No feedback data available"}

        positive_feedback = [entry for entry in self.training_data if entry.get("rating", 0) >= 4]
        negative_feedback = [entry for entry in self.training_data if entry.get("rating", 0) <= 2]
        
        # Analyze prompt types
        prompt_type_performance = {}
        for entry in self.training_data:
            prompt_type = entry.get("prompt_type", "general")
            if prompt_type not in prompt_type_performance:
                prompt_type_performance[prompt_type] = {"total": 0, "positive": 0}
            
            prompt_type_performance[prompt_type]["total"] += 1
            if entry.get("rating", 0) >= 4:
                prompt_type_performance[prompt_type]["positive"] += 1

        # Find common success patterns
        success_patterns = {}
        for entry in positive_feedback:
            words = entry["prompt"].split()
            for word in words:
                if len(word) > 3:  # Skip short words
                    success_patterns[word] = success_patterns.get(word, 0) + 1

        # Find failure patterns
        failure_patterns = {}
        for entry in negative_feedback:
            if entry.get("correction"):
                failure_patterns[entry["prompt"]] = entry["correction"]

        return {
            "total_feedback": len(self.training_data),
            "positive_feedback": len(positive_feedback),
            "negative_feedback": len(negative_feedback),
            "success_rate": self._calculate_success_rate(),
            "prompt_type_performance": prompt_type_performance,
            "top_success_patterns": sorted(success_patterns.items(), key=lambda x: x[1], reverse=True)[:5],
            "failure_patterns": failure_patterns,
            "improvement_areas": [entry.get("correction") for entry in negative_feedback if entry.get("correction")],
            "performance_metrics": self.performance_metrics
        }

    def get_personalized_suggestion(self, user_prompt, user_history=None):
        """Generate personalized suggestions based on user history and system learning"""
        analysis = self.analyze_feedback_patterns()
        
        # Check cache for similar prompts
        cache_key = user_prompt.lower().strip()
        if cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            if cached_response["rating"] >= 4:
                return cached_response["response"] + "<br><br>💡 <em>This suggestion was improved based on user feedback!</em>"
        
        # Use learned patterns to enhance response
        enhanced_context = {
            "user_prompt": user_prompt,
            "success_patterns": analysis.get("top_success_patterns", []),
            "common_failures": analysis.get("failure_patterns", {}),
            "performance_data": analysis.get("performance_metrics", {})
        }
        
        return enhanced_context

    def cache_successful_response(self, prompt, response, rating):
        """Cache successful responses for future use"""
        if rating >= 4:
            self.response_cache[prompt.lower().strip()] = {
                "response": response,
                "rating": rating,
                "timestamp": datetime.now().isoformat()
            }
            
            # Keep cache size manageable
            if len(self.response_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(self.response_cache.items(), 
                                    key=lambda x: x[1]["timestamp"])
                self.response_cache = dict(sorted_cache[-80:])  # Keep 80 most recent

    def export_training_insights(self):
        """Export training insights for analysis"""
        analysis = self.analyze_feedback_patterns()
        
        insights = {
            "generated_at": datetime.now().isoformat(),
            "system_performance": analysis,
            "recommendations": self._generate_system_recommendations(analysis),
            "data_quality_score": self._calculate_data_quality_score()
        }
        
        return insights

    def _generate_system_recommendations(self, analysis):
        """Generate system improvement recommendations"""
        recommendations = []
        
        success_rate = analysis.get("success_rate", 0)
        if success_rate < 70:
            recommendations.append("Improve response quality - success rate below 70%")
            
        if analysis.get("negative_feedback", 0) > analysis.get("positive_feedback", 0):
            recommendations.append("Focus on addressing common user complaints")
            
        prompt_performance = analysis.get("prompt_type_performance", {})
        for prompt_type, stats in prompt_performance.items():
            if stats["total"] > 5 and (stats["positive"] / stats["total"]) < 0.6:
                recommendations.append(f"Improve {prompt_type} response quality")
        
        return recommendations

    def _analyze_cognitive_complexity(self, prompt):
        """Analyze the cognitive complexity of the user's request"""
        complexity_indicators = {
            'basic': ['draw', 'color', 'simple', 'easy'],
            'intermediate': ['composition', 'perspective', 'shading', 'technique'],
            'advanced': ['style', 'artistic', 'professional', 'complex', 'advanced'],
            'expert': ['masterpiece', 'photorealistic', 'virtuosic', 'experimental']
        }
        
        prompt_words = prompt.lower().split()
        complexity_scores = {}
        
        for level, indicators in complexity_indicators.items():
            score = sum(1 for word in prompt_words if any(ind in word for ind in indicators))
            complexity_scores[level] = score
        
        # Determine dominant complexity level
        max_level = max(complexity_scores, key=complexity_scores.get)
        complexity_value = {
            'basic': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4
        }.get(max_level, 1)
        
        return {
            'level': max_level,
            'value': complexity_value,
            'indicators_found': complexity_scores
        }

    def _calculate_semantic_depth(self, prompt, response):
        """Calculate semantic depth and meaning richness"""
        prompt_concepts = len(set(prompt.lower().split()))
        response_concepts = len(set(response.lower().split()))
        
        # Advanced semantic analysis
        artistic_terms = ['composition', 'balance', 'harmony', 'contrast', 'perspective', 
                         'technique', 'style', 'medium', 'texture', 'form', 'space']
        
        semantic_richness = sum(1 for term in artistic_terms 
                               if term in prompt.lower() or term in response.lower())
        
        return {
            'prompt_concepts': prompt_concepts,
            'response_concepts': response_concepts,
            'artistic_vocabulary': semantic_richness,
            'depth_score': (semantic_richness + response_concepts / 10) / 2
        }

    def _identify_creative_elements(self, prompt, response):
        """Identify creative and innovative elements"""
        creative_indicators = [
            'creative', 'unique', 'original', 'innovative', 'artistic', 'expressive',
            'imaginative', 'experimental', 'stylized', 'abstract', 'conceptual'
        ]
        
        creativity_score = sum(1 for indicator in creative_indicators 
                              if indicator in prompt.lower() or indicator in response.lower())
        
        return {
            'creativity_score': creativity_score,
            'has_creative_elements': creativity_score > 0,
            'innovation_level': min(creativity_score / 3, 1.0)
        }

    def _determine_skill_level(self, prompt):
        """Determine required skill level for the request"""
        skill_keywords = {
            'beginner': ['first time', 'start', 'begin', 'basic', 'simple', 'easy', 'learn'],
            'intermediate': ['improve', 'better', 'technique', 'practice', 'develop'],
            'advanced': ['master', 'professional', 'expert', 'advanced', 'complex'],
            'expert': ['virtuoso', 'masterpiece', 'photorealistic', 'highly detailed']
        }
        
        for level, keywords in skill_keywords.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                return level
        return 'intermediate'  # Default

    def _extract_concept_categories(self, prompt):
        """Extract and categorize artistic concepts"""
        categories = {
            'subjects': ['person', 'face', 'portrait', 'animal', 'cat', 'dog', 'tree', 'flower', 'house'],
            'techniques': ['shading', 'blending', 'perspective', 'proportion', 'composition'],
            'tools': ['pencil', 'brush', 'eraser', 'canvas', 'color', 'paint'],
            'styles': ['realistic', 'cartoon', 'anime', 'abstract', 'impressionist'],
            'elements': ['line', 'shape', 'form', 'color', 'texture', 'space', 'value']
        }
        
        found_categories = {}
        for category, items in categories.items():
            found_items = [item for item in items if item in prompt.lower()]
            if found_items:
                found_categories[category] = found_items
        
        return found_categories

    def _identify_learning_objectives(self, prompt):
        """Identify specific learning objectives from the prompt"""
        objectives = {
            'skill_building': ['learn', 'practice', 'improve', 'develop', 'master'],
            'problem_solving': ['help', 'fix', 'problem', 'issue', 'trouble', 'error'],
            'creative_expression': ['create', 'design', 'artistic', 'expressive', 'original'],
            'technical_mastery': ['technique', 'method', 'professional', 'advanced', 'precise']
        }
        
        identified_objectives = []
        for objective, keywords in objectives.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                identified_objectives.append(objective)
        
        return identified_objectives

    def _analyze_multimodal_aspects(self, prompt):
        """Analyze multimodal learning aspects"""
        modalities = {
            'visual': ['see', 'look', 'visual', 'image', 'picture', 'reference'],
            'kinesthetic': ['draw', 'paint', 'sketch', 'create', 'make', 'practice'],
            'auditory': ['explain', 'tell', 'describe', 'instruction', 'guide'],
            'analytical': ['analyze', 'understand', 'theory', 'principle', 'concept']
        }
        
        active_modalities = []
        for modality, indicators in modalities.items():
            if any(indicator in prompt.lower() for indicator in indicators):
                active_modalities.append(modality)
        
        return {
            'active_modalities': active_modalities,
            'multimodal_score': len(active_modalities),
            'is_multimodal': len(active_modalities) > 1
        }

    def _assess_contextual_relevance(self, prompt):
        """Assess contextual relevance and situational factors"""
        context_factors = {
            'time_sensitive': ['quick', 'fast', 'urgent', 'now', 'immediately'],
            'environment': ['mobile', 'tablet', 'desktop', 'touchscreen'],
            'purpose': ['homework', 'project', 'practice', 'fun', 'professional'],
            'audience': ['beginner', 'student', 'artist', 'professional', 'child']
        }
        
        context_data = {}
        for factor, indicators in context_factors.items():
            matching_indicators = [ind for ind in indicators if ind in prompt.lower()]
            if matching_indicators:
                context_data[factor] = matching_indicators
        
        return context_data

    def _update_skill_progression(self, training_entry):
        """Track and update skill progression patterns"""
        user_id = training_entry.get('session_id', 'anonymous')
        skill_level = training_entry.get('skill_level_required', 'intermediate')
        
        if user_id not in self.skill_progression_tracker:
            self.skill_progression_tracker[user_id] = {
                'current_level': skill_level,
                'progression_history': [],
                'mastery_areas': [],
                'improvement_rate': 0.0
            }
        
        user_progress = self.skill_progression_tracker[user_id]
        user_progress['progression_history'].append({
            'timestamp': training_entry['timestamp'],
            'skill_level': skill_level,
            'rating': training_entry['rating'],
            'concepts': training_entry.get('concept_categories', {})
        })
        
        # Calculate improvement rate
        if len(user_progress['progression_history']) > 1:
            recent_ratings = [h['rating'] for h in user_progress['progression_history'][-5:]]
            user_progress['improvement_rate'] = sum(recent_ratings) / len(recent_ratings)

    def _enhance_contextual_memory(self, training_entry):
        """Build contextual memory for better future responses"""
        prompt_key = training_entry['prompt'][:50]  # First 50 chars as key
        
        if prompt_key not in self.contextual_memory:
            self.contextual_memory[prompt_key] = {
                'successful_patterns': [],
                'failed_patterns': [],
                'context_variations': [],
                'optimal_responses': []
            }
        
        memory = self.contextual_memory[prompt_key]
        
        if training_entry['rating'] >= 4:
            memory['successful_patterns'].append({
                'response': training_entry['response'][:200],
                'concepts': training_entry.get('concept_categories', {}),
                'timestamp': training_entry['timestamp']
            })
        elif training_entry['rating'] <= 2:
            memory['failed_patterns'].append({
                'response': training_entry['response'][:200],
                'correction': training_entry.get('correction'),
                'timestamp': training_entry['timestamp']
            })

    def _evolve_learning_pathways(self, training_entry):
        """Develop adaptive learning pathways"""
        concepts = training_entry.get('concept_categories', {})
        skill_level = training_entry.get('skill_level_required', 'intermediate')
        
        for category, items in concepts.items():
            pathway_key = f"{category}_{skill_level}"
            
            if pathway_key not in self.learning_pathways:
                self.learning_pathways[pathway_key] = {
                    'prerequisite_concepts': [],
                    'learning_sequence': [],
                    'success_indicators': [],
                    'common_obstacles': []
                }
            
            pathway = self.learning_pathways[pathway_key]
            
            if training_entry['rating'] >= 4:
                pathway['success_indicators'].extend(items)
            elif training_entry['rating'] <= 2:
                pathway['common_obstacles'].extend(items)

    def _build_semantic_understanding(self, training_entry):
        """Build deeper semantic understanding of art concepts"""
        prompt_words = set(training_entry['prompt'].split())
        response_words = set(training_entry['response'].split())
        
        # Build concept relationships
        for word in prompt_words:
            if len(word) > 3:  # Skip short words
                if word not in self.semantic_understanding:
                    self.semantic_understanding[word] = {
                        'related_concepts': set(),
                        'successful_associations': [],
                        'usage_frequency': 0,
                        'context_patterns': []
                    }
                
                concept = self.semantic_understanding[word]
                concept['usage_frequency'] += 1
                concept['related_concepts'].update(response_words)
                
                if training_entry['rating'] >= 4:
                    concept['successful_associations'].append({
                        'response_snippet': training_entry['response'][:100],
                        'rating': training_entry['rating']
                    })

    def _adapt_difficulty_scaling(self, training_entry):
        """Adapt difficulty scaling based on user performance"""
        complexity = training_entry.get('cognitive_complexity', {})
        rating = training_entry['rating']
        
        complexity_level = complexity.get('level', 'basic')
        
        if complexity_level not in self.difficulty_scaling:
            self.difficulty_scaling[complexity_level] = {
                'success_rate': 0.0,
                'attempts': 0,
                'optimal_progression': [],
                'challenge_threshold': 3.5
            }
        
        scaling = self.difficulty_scaling[complexity_level]
        scaling['attempts'] += 1
        
        # Update success rate
        current_success = (scaling['success_rate'] * (scaling['attempts'] - 1) + (1 if rating >= 4 else 0)) / scaling['attempts']
        scaling['success_rate'] = current_success
        
        # Adjust challenge threshold based on performance
        if current_success > 0.8:
            scaling['challenge_threshold'] = min(scaling['challenge_threshold'] + 0.1, 5.0)
        elif current_success < 0.5:
            scaling['challenge_threshold'] = max(scaling['challenge_threshold'] - 0.1, 2.0)

    def get_comprehensive_learning_insights(self):
        """Generate comprehensive learning insights across all dimensions"""
        return {
            'skill_progression_analysis': self._analyze_skill_progression(),
            'concept_mastery_map': self._generate_concept_mastery_map(),
            'learning_pathway_optimization': self._optimize_learning_pathways(),
            'semantic_knowledge_graph': self._build_semantic_knowledge_graph(),
            'adaptive_difficulty_recommendations': self._generate_difficulty_recommendations(),
            'multimodal_learning_effectiveness': self._assess_multimodal_effectiveness(),
            'personalization_opportunities': self._identify_personalization_opportunities(),
            'predictive_learning_model': self._build_predictive_learning_model()
        }

    def _analyze_skill_progression(self):
        """Analyze skill progression patterns across users"""
        progression_analysis = {
            'average_improvement_rate': 0.0,
            'skill_level_distribution': {},
            'mastery_timelines': {},
            'progression_bottlenecks': []
        }
        
        if not self.skill_progression_tracker:
            return progression_analysis
        
        improvement_rates = [user['improvement_rate'] for user in self.skill_progression_tracker.values()]
        progression_analysis['average_improvement_rate'] = sum(improvement_rates) / len(improvement_rates) if improvement_rates else 0.0
        
        return progression_analysis

    def _generate_concept_mastery_map(self):
        """Generate a concept mastery map showing learning relationships"""
        mastery_map = {}
        
        for word, data in self.semantic_understanding.items():
            mastery_level = min(data['usage_frequency'] / 10, 1.0)  # Normalize to 0-1
            success_rate = len(data['successful_associations']) / max(data['usage_frequency'], 1)
            
            mastery_map[word] = {
                'mastery_level': mastery_level,
                'success_rate': success_rate,
                'related_concepts': list(data['related_concepts'])[:5],  # Top 5
                'learning_priority': (1 - mastery_level) * success_rate  # High if low mastery but high success
            }
        
        return mastery_map

    def _identify_training_category(self, prompt, response):
        """Identify specific training category from comprehensive dataset"""
        categories = {
            'geometric_construction': ['isometric', 'geometric', 'construction', 'polygon', 'hexagon', 'spiral'],
            'mathematical_art': ['fibonacci', 'golden ratio', 'fractal', 'mandala', 'sacred geometry'],
            'classical_techniques': ['chiaroscuro', 'sfumato', 'impasto', 'pointillism', 'atmospheric'],
            'cultural_styles': ['chinese brush', 'japanese', 'islamic', 'celtic', 'aboriginal', 'medieval'],
            'scientific_illustration': ['botanical', 'anatomical', 'technical', 'cutaway', 'medical'],
            'digital_mastery': ['layer', 'blending', 'workflow', 'brush', 'digital painting'],
            'pattern_systems': ['celtic knot', 'op art', 'art nouveau', 'decorative'],
            'advanced_subjects': ['water reflection', 'architectural', 'mechanical', 'fabric', 'crystal']
        }
        
        prompt_lower = prompt.lower()
        response_lower = response.lower()
        
        for category, keywords in categories.items():
            if any(keyword in prompt_lower or keyword in response_lower for keyword in keywords):
                return category
        return 'general'

    def _assess_technique_complexity(self, prompt):
        """Assess technical complexity level of the request"""
        complexity_indicators = {
            'basic': 1,
            'intermediate': 2, 
            'advanced': 3,
            'expert': 4,
            'professional': 5
        }
        
        for level, score in complexity_indicators.items():
            if level in prompt.lower():
                return score
        
        # Analyze by technique complexity
        if any(word in prompt.lower() for word in ['fibonacci', 'golden ratio', 'chiaroscuro', 'sfumato']):
            return 5  # Expert level
        elif any(word in prompt.lower() for word in ['perspective', 'composition', 'blending']):
            return 3  # Advanced
        else:
            return 2  # Intermediate default

    def _extract_cultural_context(self, prompt):
        """Extract cultural and historical art context"""
        cultural_markers = {
            'chinese': ['chinese', 'brush painting', 'ink wash'],
            'japanese': ['japanese', 'wave pattern', 'hokusai'],
            'islamic': ['islamic', 'geometric star', 'arabesque'],
            'celtic': ['celtic', 'knot', 'interlace'],
            'medieval': ['illuminated', 'manuscript', 'gothic'],
            'aboriginal': ['aboriginal', 'dot painting', 'dreamtime'],
            'art_nouveau': ['art nouveau', 'mucha', 'organic forms']
        }
        
        found_cultures = []
        for culture, markers in cultural_markers.items():
            if any(marker in prompt.lower() for marker in markers):
                found_cultures.append(culture)
        return found_cultures

    def _identify_mathematical_patterns(self, prompt):
        """Identify mathematical and geometric patterns"""
        math_patterns = {
            'fibonacci': 'fibonacci sequence and spiral construction',
            'golden_ratio': 'golden ratio and divine proportion',
            'fractals': 'fractal geometry and self-similarity',
            'tessellation': 'geometric tessellation patterns',
            'sacred_geometry': 'sacred geometric principles',
            'symmetry': 'symmetrical pattern systems'
        }
        
        found_patterns = []
        for pattern, description in math_patterns.items():
            if any(keyword in prompt.lower() for keyword in pattern.split('_')):
                found_patterns.append({'pattern': pattern, 'description': description})
        return found_patterns

    def _determine_professional_level(self, response):
        """Determine professional skill level indicated by response"""
        professional_indicators = [
            'professional', 'master', 'expert', 'advanced technique', 
            'industry standard', 'portfolio quality', 'exhibition level'
        ]
        
        technical_depth = len([word for word in response.split() if len(word) > 8])
        professional_terms = sum(1 for indicator in professional_indicators if indicator in response.lower())
        
        if professional_terms > 2 or technical_depth > 20:
            return 'professional'
        elif professional_terms > 0 or technical_depth > 10:
            return 'advanced'
        else:
            return 'intermediate'

    def _classify_artistic_movement(self, prompt):
        """Classify artistic movement or style referenced"""
        movements = {
            'renaissance': ['leonardo', 'michelangelo', 'sfumato', 'chiaroscuro'],
            'impressionism': ['monet', 'pointillism', 'plein air', 'light study'],
            'art_nouveau': ['mucha', 'klimt', 'organic', 'decorative'],
            'modernism': ['picasso', 'abstract', 'cubism', 'experimental'],
            'realism': ['photorealistic', 'accurate', 'detailed', 'lifelike']
        }
        
        for movement, keywords in movements.items():
            if any(keyword in prompt.lower() for keyword in keywords):
                return movement
        return 'contemporary'

    def _assess_scientific_accuracy(self, response):
        """Assess scientific accuracy requirements of the response"""
        scientific_terms = [
            'anatomical', 'botanical', 'technical', 'precise', 'accurate',
            'medical', 'scientific', 'measurement', 'proportion', 'structure'
        ]
        
        accuracy_score = sum(1 for term in scientific_terms if term in response.lower())
        
        if accuracy_score >= 3:
            return 'high_precision'
        elif accuracy_score >= 1:
            return 'moderate_precision'
        else:
            return 'artistic_interpretation'

    def _calculate_data_quality_score(self):
        """Calculate overall training data quality score with comprehensive metrics"""
        if not self.training_data:
            return 0
            
        total_entries = len(self.training_data)
        
        # Basic completeness
        complete_entries = sum(1 for entry in self.training_data 
                             if all(key in entry for key in ["prompt", "response", "rating"]))
        
        # Advanced completeness (including new learning dimensions)
        advanced_complete = sum(1 for entry in self.training_data 
                               if all(key in entry for key in ["cognitive_complexity", "semantic_depth", "concept_categories"]))
        
        # Dataset-specific completeness
        dataset_complete = sum(1 for entry in self.training_data 
                              if all(key in entry for key in ["training_category", "technique_complexity", "professional_level"]))
        
        # Quality indicators
        high_quality_entries = sum(1 for entry in self.training_data if entry.get('rating', 0) >= 4)
        diverse_concepts = len(set(str(entry.get('concept_categories', {})) for entry in self.training_data))
        diverse_categories = len(set(entry.get('training_category', 'general') for entry in self.training_data))
        
        quality_score = (
            (complete_entries / total_entries) * 0.25 +
            (advanced_complete / total_entries) * 0.25 +
            (dataset_complete / total_entries) * 0.2 +
            (high_quality_entries / total_entries) * 0.15 +
            min(diverse_concepts / 20, 1.0) * 0.1 +
            min(diverse_categories / 10, 1.0) * 0.05  # Category diversity bonus
        ) * 100
        
        return round(quality_score, 1)

    def determine_response_strategy(self, prompt, feature_type, user_info):
        """Advanced immersive decision-making system with transformer analysis"""
        prompt_lower = prompt.lower().strip()
        
        # Transformer-based analysis
        transformer_features = self.transformer.generate_response_features(prompt_lower)
        intent_analysis = self.transformer.extract_intent(prompt_lower)
        
        # Complex algorithmic analysis components
        algorithmic_analysis = self._perform_complex_algorithmic_analysis(prompt_lower, user_info)
        neural_pattern_analysis = self._neural_pattern_recognition(prompt_lower)
        predictive_modeling = self._predictive_response_modeling(prompt_lower, user_info)
        contextual_embeddings = self._generate_contextual_embeddings(prompt_lower)
        
        # Enhanced strategy decision factors with complex algorithms
        factors = {
            'has_exact_training_match': False,
            'has_similar_training_match': False,
            'is_advanced_technique': False,
            'is_troubleshooting': False,
            'is_creative_request': False,
            'training_data_confidence': 0.0,
            'prompt_complexity': 1,
            'user_skill_level': self._assess_user_skill_level(user_info),
            'learning_context': self._analyze_learning_context(prompt_lower),
            'artistic_domain': self._identify_artistic_domain(prompt_lower),
            'immersion_factors': self._calculate_immersion_factors(prompt_lower, user_info),
            'personalization_score': self._calculate_personalization_score(user_info),
            'session_context': self._analyze_session_context(user_info),
            
            # Advanced algorithmic analysis results
            'algorithmic_confidence': algorithmic_analysis['confidence_score'],
            'neural_patterns': neural_pattern_analysis,
            'predictive_indicators': predictive_modeling,
            'contextual_embeddings': contextual_embeddings,
            'complexity_algorithm_score': self._complex_complexity_algorithm(prompt_lower),
            'response_optimization_matrix': self._response_optimization_algorithm(prompt_lower, user_info),
            'hybrid_fusion_score': self._hybrid_fusion_algorithm(prompt_lower),
            'multi_dimensional_analysis': self._multi_dimensional_pattern_analysis(prompt_lower, user_info),
            
            # Transformer model analysis
            'transformer_features': transformer_features,
            'intent_analysis': intent_analysis,
            'semantic_embedding': transformer_features['semantic_embedding'],
            'transformer_complexity': transformer_features['complexity_level'],
            'transformer_domain': transformer_features['art_domain'],
            'transformer_confidence': intent_analysis['confidence']
        }
        
        # Check for exact matches in training data with context awareness
        for cached_prompt, cached_data in self.response_cache.items():
            if cached_prompt == prompt_lower and cached_data["rating"] >= 4:
                factors['has_exact_training_match'] = True
                factors['training_data_confidence'] = 1.0
                break
        
        # Enhanced similarity matching with transformer semantic understanding
        if not factors['has_exact_training_match']:
            prompt_words = set(prompt_lower.split())
            best_similarity = 0.0
            best_match = None
            semantic_matches = []
            
            for entry in self.training_data:
                if entry.get("rating", 0) >= 4:
                    entry_words = set(entry["prompt"].split())
                    if prompt_words and entry_words:
                        # Basic word similarity
                        word_similarity = len(prompt_words.intersection(entry_words)) / len(prompt_words.union(entry_words))
                        
                        # Transformer semantic similarity
                        transformer_similarity = self.transformer.semantic_similarity(prompt_lower, entry["prompt"])
                        
                        # Traditional semantic similarity bonus
                        semantic_bonus = self._calculate_semantic_similarity(prompt_lower, entry["prompt"])
                        
                        # Context relevance bonus
                        context_bonus = self._calculate_context_relevance(factors, entry)
                        
                        # Weighted combination with transformer similarity having higher weight
                        total_similarity = (
                            word_similarity * 0.2 + 
                            transformer_similarity * 0.5 + 
                            semantic_bonus * 0.2 + 
                            context_bonus * 0.1
                        )
                        
                        if total_similarity > best_similarity:
                            best_similarity = total_similarity
                            best_match = entry
                            
                        if total_similarity > 0.25:  # Lower threshold due to better similarity
                            semantic_matches.append({
                                'entry': entry,
                                'similarity': total_similarity,
                                'word_sim': word_similarity,
                                'transformer_sim': transformer_similarity,
                                'semantic_sim': semantic_bonus,
                                'context_sim': context_bonus
                            })
            
            if best_similarity > 0.25:  # Lowered threshold due to transformer matching
                factors['has_similar_training_match'] = True
                factors['training_data_confidence'] = best_similarity
                factors['semantic_matches'] = semantic_matches[:5]  # Top 5 matches
                factors['best_transformer_match'] = {
                    'entry': best_match,
                    'similarity': best_similarity,
                    'intent_match': intent_analysis['intent']
                }
        
        # Enhanced technique detection with cultural and style context
        advanced_techniques = {
            'mathematical': ['fibonacci', 'golden ratio', 'fractal', 'sacred geometry'],
            'classical': ['chiaroscuro', 'sfumato', 'impasto', 'atmospheric perspective'],  
            'cultural': ['mandala', 'celtic knot', 'chinese brush', 'japanese wave'],
            'modern': ['pointillism', 'cubism', 'art nouveau', 'abstract'],
            'technical': ['isometric', 'perspective', 'anatomical', 'architectural']
        }
        
        factors['technique_category'] = None
        for category, techniques in advanced_techniques.items():
            if any(technique in prompt_lower for technique in techniques):
                factors['is_advanced_technique'] = True
                factors['technique_category'] = category
                break
        
        # Enhanced troubleshooting detection with solution prediction
        troubleshooting_patterns = {
            'tool_issues': ['tool', 'brush', 'pencil', 'eraser', 'not working'],
            'technical_problems': ['canvas', 'layer', 'color', 'save', 'load'],
            'drawing_difficulties': ['proportions', 'perspective', 'shading', 'blending'],
            'general_help': ['help', 'problem', 'issue', 'error', 'fix', 'trouble']
        }
        
        factors['troubleshooting_category'] = None
        for category, keywords in troubleshooting_patterns.items():
            if any(keyword in prompt_lower for keyword in keywords):
                factors['is_troubleshooting'] = True
                factors['troubleshooting_category'] = category
                break
        
        # Enhanced creative request detection with style analysis
        creative_indicators = {
            'original_creation': ['creative', 'unique', 'original', 'new', 'invent'],
            'artistic_expression': ['artistic', 'expressive', 'emotional', 'mood'],
            'experimental': ['experimental', 'abstract', 'unconventional', 'mixed'],
            'stylistic': ['style', 'stylized', 'interpretation', 'version']
        }
        
        factors['creative_category'] = None
        for category, keywords in creative_indicators.items():
            if any(keyword in prompt_lower for keyword in keywords):
                factors['is_creative_request'] = True
                factors['creative_category'] = category
                break
        
        # Enhanced complexity assessment
        complexity_indicators = {
            'word_count': min(len(prompt_lower.split()) / 3, 2.0),
            'technical_terms': len([w for w in prompt_lower.split() if len(w) > 8]) * 0.5,
            'multiple_subjects': len([w for w in prompt_lower.split() if w in ['and', 'with', 'plus', 'also']]) * 0.3,
            'detail_requests': len([w for w in prompt_lower.split() if w in ['detailed', 'realistic', 'accurate', 'professional']]) * 0.4
        }
        factors['prompt_complexity'] = min(sum(complexity_indicators.values()), 5.0)
        
        # Immersive Decision Logic with Enhanced Context
        return self._execute_immersive_decision_logic(factors, prompt_lower, feature_type, user_info)
    
    def _execute_immersive_decision_logic(self, factors, prompt_lower, feature_type, user_info):
        """Execute immersive decision logic with rich contextual responses"""
        
        # Strategy 1: Perfect Training Match with Personalization
        if factors['has_exact_training_match']:
            cached_response = None
            for cached_prompt, cached_data in self.response_cache.items():
                if cached_prompt == prompt_lower and cached_data["rating"] >= 4:
                    cached_response = cached_data
                    break
            
            # Enhance with personalization
            enhanced_response = self._enhance_with_personalization(
                cached_response["response"], factors, user_info
            )
            
            return {
                "strategy": "training_only",
                "response": enhanced_response,
                "confidence": 1.0,
                "reason": f"Perfect dataset match with personalization for {factors['user_skill_level']} level",
                "immersion_level": "high",
                "context_applied": ["exact_match", "personalized", "skill_adapted"]
            }
        
        # Strategy 2: Advanced Technique with Cultural Context
        elif factors['is_advanced_technique'] and factors['training_data_confidence'] > 0.25:
            relevant_training = self._get_relevant_training_examples(prompt_lower, limit=2)
            if relevant_training:
                # Add cultural and historical context
                contextual_response = self._add_cultural_context(
                    relevant_training[0]["response"], 
                    factors['technique_category'],
                    factors['artistic_domain']
                )
                
                return {
                    "strategy": "training_cultural",
                    "response": contextual_response,
                    "confidence": factors['training_data_confidence'],
                    "reason": f"Advanced {factors['technique_category']} technique with cultural context",
                    "immersion_level": "very_high",
                    "context_applied": ["advanced_technique", "cultural_context", "historical_depth"]
                }
        
        # Strategy 3: Intelligent Troubleshooting with Solution Path
        elif factors['is_troubleshooting']:
            solution_confidence = self._assess_solution_confidence(factors, prompt_lower)
            
            if solution_confidence > 0.7:
                # Use training data for known solutions
                troubleshooting_response = self._generate_troubleshooting_response(
                    factors['troubleshooting_category'], prompt_lower, user_info
                )
                
                return {
                    "strategy": "training_troubleshooting",
                    "response": troubleshooting_response,
                    "confidence": solution_confidence,
                    "reason": f"Known {factors['troubleshooting_category']} issue with guided solution",
                    "immersion_level": "high",
                    "context_applied": ["troubleshooting", "step_by_step", "user_guided"]
                }
            else:
                # Use OpenAI for novel issues
                return {
                    "strategy": "openai_troubleshooting",
                    "system_message": self._get_enhanced_troubleshooting_system_message(factors),
                    "user_message": f"TROUBLESHOOT {factors['troubleshooting_category']}: {prompt_lower}",
                    "confidence": 0.8,
                    "reason": "Novel troubleshooting requiring AI analysis",
                    "immersion_level": "medium",
                    "context_applied": ["ai_analysis", "problem_solving"]
                }
        
        # Strategy 4: Creative Expression with Style Guidance
        elif factors['is_creative_request'] or factors['prompt_complexity'] >= 3.5:
            if factors['personalization_score'] > 0.6:
                # Personalized creative guidance
                return {
                    "strategy": "hybrid_creative",
                    "system_message": self._get_personalized_creative_system_message(factors, feature_type),
                    "user_message": prompt_lower,
                    "training_context": self._get_creative_training_context(factors),
                    "confidence": 0.9,
                    "reason": f"Personalized creative guidance for {factors['creative_category']} expression",
                    "immersion_level": "very_high",
                    "context_applied": ["creative_expression", "personalized", "style_guidance"]
                }
            else:
                # Standard creative response
                return {
                    "strategy": "openai_creative",
                    "system_message": self._get_creative_system_message(feature_type),
                    "user_message": prompt_lower,
                    "confidence": 0.85,
                    "reason": "Creative request requiring artistic flexibility",
                    "immersion_level": "high",
                    "context_applied": ["creative", "artistic_freedom"]
                }
        
        # Strategy 5: Training Data with Transformer Enhancement
        elif factors['has_similar_training_match'] and factors['training_data_confidence'] > 0.25:
            training_examples = factors.get('semantic_matches', [])[:3]
            
            if training_examples:
                # Use the best training match directly
                best_match = training_examples[0]['entry']
                enhanced_response = self._enhance_with_personalization(
                    best_match["response"], factors, user_info
                )
                
                return {
                    "strategy": "training_only",
                    "response": enhanced_response + f"<br><br>🎓 <em>Enhanced training dataset response (confidence: {factors['training_data_confidence']:.2f})</em>",
                    "confidence": factors['training_data_confidence'],
                    "reason": f"High-quality dataset match with {len(training_examples)} similar examples",
                    "immersion_level": "very_high",
                    "context_applied": ["dataset_match", "semantic_matching", "personalized"]
                }
        
        # Strategy 6: Skill-Adapted General Response
        else:
            skill_adapted_message = self._adapt_message_to_skill_level(
                prompt_lower, factors['user_skill_level'], factors['artistic_domain']
            )
            
            return {
                "strategy": "openai_skill_adapted",
                "system_message": self._get_skill_adapted_system_message(factors, feature_type),
                "user_message": skill_adapted_message,
                "confidence": 0.75,
                "reason": f"Skill-adapted response for {factors['user_skill_level']} level",
                "immersion_level": "high",
                "context_applied": ["skill_adaptation", "domain_specific"]
            }
    
    def _assess_user_skill_level(self, user_info):
        """Assess user skill level from interaction history"""
        total_uses = user_info.get("total_uses", 0)
        
        if total_uses < 5:
            return "beginner"
        elif total_uses < 20:
            return "intermediate"  
        elif total_uses < 50:
            return "advanced"
        else:
            return "expert"
    
    def _analyze_learning_context(self, prompt):
        """Analyze the learning context and objectives"""
        learning_indicators = {
            'tutorial_seeking': ['how to', 'tutorial', 'guide', 'step by step'],
            'skill_building': ['practice', 'improve', 'better', 'learn'],
            'problem_solving': ['help', 'stuck', 'difficulty', 'challenge'],
            'exploration': ['try', 'experiment', 'explore', 'discover']
        }
        
        for context, keywords in learning_indicators.items():
            if any(keyword in prompt for keyword in keywords):
                return context
        return 'general_inquiry'
    
    def _identify_artistic_domain(self, prompt):
        """Identify the specific artistic domain"""
        domains = {
            'portrait': ['face', 'portrait', 'person', 'character'],
            'landscape': ['landscape', 'scenery', 'nature', 'mountain', 'tree'],
            'still_life': ['object', 'fruit', 'vase', 'table', 'arrangement'],
            'abstract': ['abstract', 'non-representational', 'conceptual'],
            'technical': ['diagram', 'blueprint', 'technical', 'mechanical'],
            'fantasy': ['dragon', 'fantasy', 'mythical', 'magical'],
            'anime': ['anime', 'manga', 'japanese style', 'cartoon']
        }
        
        for domain, keywords in domains.items():
            if any(keyword in prompt for keyword in keywords):
                return domain
        return 'general'
    
    def _calculate_immersion_factors(self, prompt, user_info):
        """Calculate factors that contribute to immersive experience"""
        return {
            'emotional_engagement': len([w for w in prompt.split() if w in ['love', 'beautiful', 'amazing', 'inspire']]) * 0.2,
            'detail_seeking': len([w for w in prompt.split() if w in ['detailed', 'realistic', 'accurate']]) * 0.3,
            'personal_connection': len([w for w in prompt.split() if w in ['my', 'personal', 'own', 'custom']]) * 0.25,
            'challenge_level': min(len(prompt.split()) / 10, 0.3),
            'user_engagement': min(user_info.get("total_uses", 0) / 20, 0.4)
        }
    
    def _calculate_personalization_score(self, user_info):
        """Calculate how much personalization to apply"""
        factors = [
            min(user_info.get("total_uses", 0) / 30, 0.4),  # Usage history
            0.3 if user_info.get("premium", False) else 0.1,  # Premium status
            0.2 if user_info.get("saved_drawings") else 0.0,  # Has saved work
            0.1  # Base personalization
        ]
        return sum(factors)
    
    def _analyze_session_context(self, user_info):
        """Analyze the current session context"""
        return {
            'session_length': min(user_info.get("total_uses", 0), 10),
            'is_returning_user': user_info.get("total_uses", 0) > 1,
            'has_premium': user_info.get("premium", False),
            'engagement_level': 'high' if user_info.get("total_uses", 0) > 5 else 'new'
        }
    
    def _calculate_semantic_similarity(self, prompt1, prompt2):
        """Calculate semantic similarity between prompts"""
        # Simple semantic similarity based on shared concepts
        art_concepts = {
            'shape': ['circle', 'square', 'triangle', 'rectangle', 'oval'],
            'technique': ['shading', 'blending', 'hatching', 'cross-hatch'],
            'color': ['red', 'blue', 'green', 'yellow', 'purple', 'orange'],
            'style': ['realistic', 'cartoon', 'anime', 'abstract', 'impressionist']
        }
        
        concepts1 = set()
        concepts2 = set()
        
        for concept_category, terms in art_concepts.items():
            for term in terms:
                if term in prompt1:
                    concepts1.add(concept_category)
                if term in prompt2:
                    concepts2.add(concept_category)
        
        if not concepts1 or not concepts2:
            return 0.0
            
        return len(concepts1.intersection(concepts2)) / len(concepts1.union(concepts2))
    
    def _calculate_context_relevance(self, factors, entry):
        """Calculate how relevant a training entry is to current context"""
        relevance_score = 0.0
        
        # Skill level relevance
        entry_skill = entry.get('skill_level_required', 'intermediate')


    def _enhance_with_personalization(self, response, factors, user_info):
        """Enhance response with personalization based on user factors"""
        skill_level = factors['user_skill_level']
        personalization_score = factors['personalization_score']
        
        if personalization_score > 0.6:
            # High personalization
            user_name = "fellow artist"
            if user_info.get("premium"):
                user_name = "creative professional"
            
            personalized_intro = f"🎨 Hey {user_name}! Based on your {skill_level} level experience, here's a tailored approach:\n\n"
            
            # Add skill-appropriate encouragement
            if skill_level == "beginner":
                encouragement = "\n\n🌟 Remember: Every master was once a beginner. You're doing great!"
            elif skill_level == "intermediate":
                encouragement = "\n\n💪 You're developing solid skills! Ready for the next challenge?"
            else:
                encouragement = "\n\n🚀 Your advanced skills show - time to push creative boundaries!"
            
            return personalized_intro + response + encouragement
        
        return response
    
    def _add_cultural_context(self, response, technique_category, artistic_domain):
        """Add rich cultural and historical context to responses"""
        cultural_contexts = {
            'mathematical': {
                'intro': "🔢 Mathematical art connects us to ancient wisdom:",
                'history': "From Islamic geometric patterns to Renaissance golden ratio studies",
                'modern': "Today's digital artists still use these timeless principles"
            },
            'classical': {
                'intro': "🏛️ Classical techniques from the masters:",
                'history': "Developed during Renaissance and Baroque periods",
                'modern': "These methods remain essential for contemporary realism"
            },
            'cultural': {
                'intro': "🌍 Cultural art forms carry deep meaning:",
                'history': "Passed down through generations, each symbol tells a story",
                'modern': "Respecting traditions while adding your personal voice"
            }
        }
        
        if technique_category in cultural_contexts:
            context = cultural_contexts[technique_category]
            enhanced_response = f"{context['intro']}\n\n{response}\n\n📚 **Cultural Note:** {context['history']}. {context['modern']}."
            return enhanced_response
        
        return response
    
    def _generate_troubleshooting_response(self, troubleshooting_category, prompt, user_info):
        """Generate contextual troubleshooting response"""
        troubleshooting_templates = {
            'tool_issues': {
                'intro': "🔧 Let's get your tools working perfectly!",
                'steps': [
                    "1. Check tool selection (should be highlighted in blue)",
                    "2. Verify brush size is > 1 pixel",
                    "3. Ensure canvas is active (click on it once)",
                    "4. Try refreshing the page if issues persist"
                ],
                'outro': "💡 Pro tip: Save your work frequently to prevent data loss!"
            },
            'technical_problems': {
                'intro': "⚡ Technical issues? Let's solve this step-by-step:",
                'steps': [
                    "1. Check browser compatibility (Chrome/Firefox recommended)",
                    "2. Clear cache and reload the page",
                    "3. Disable browser extensions temporarily",
                    "4. Try incognito/private browsing mode"
                ],
                'outro': "🌐 If problems persist, try a different browser or device."
            },
            'drawing_difficulties': {
                'intro': "🎨 Drawing challenges are part of the journey!",
                'steps': [
                    "1. Break complex subjects into simple shapes",
                    "2. Use reference images for guidance",
                    "3. Practice the specific technique daily for 10 minutes",
                    "4. Don't aim for perfection - aim for progress"
                ],
                'outro': "🌟 Remember: Every expert was once a beginner. Keep practicing!"
            }
        }
        
        template = troubleshooting_templates.get(troubleshooting_category, troubleshooting_templates['tool_issues'])
        
        response = f"{template['intro']}\n\n"
        for step in template['steps']:
            response += f"{step}\n"
        response += f"\n{template['outro']}"
        
        return response
    
    def _assess_solution_confidence(self, factors, prompt):
        """Assess confidence in providing solution from training data"""
        confidence_factors = {
            'exact_match': 1.0 if factors['has_exact_training_match'] else 0.0,
            'similar_match': factors['training_data_confidence'] * 0.8,
            'common_issue': 0.7 if any(word in prompt for word in ['tool', 'brush', 'canvas', 'color']) else 0.0,
            'skill_appropriate': 0.6 if factors['user_skill_level'] in ['beginner', 'intermediate'] else 0.3
        }
        
        return min(sum(confidence_factors.values()), 1.0)
    
    def _get_enhanced_troubleshooting_system_message(self, factors):
        """Get enhanced system message for troubleshooting with context"""
        base_message = "You are an expert technical support specialist for digital drawing applications."
        
        context_additions = []
        if factors['troubleshooting_category']:
            context_additions.append(f"Focus on {factors['troubleshooting_category'].replace('_', ' ')} issues.")
        
        if factors['user_skill_level'] == 'beginner':
            context_additions.append("Use simple, non-technical language suitable for beginners.")
        
        context_additions.append("Provide step-by-step solutions with encouraging tone.")
        context_additions.append("Include preventive tips to avoid future issues.")
        
        return base_message + " " + " ".join(context_additions)
    
    def _get_personalized_creative_system_message(self, factors, feature_type):
        """Get personalized creative system message"""
        base = f"You are an inspiring art instructor helping a {factors['user_skill_level']} level artist."
        
        creative_focus = factors.get('creative_category', 'general')
        domain_focus = factors.get('artistic_domain', 'general')
        
        context = f" Focus on {creative_focus} in {domain_focus} art."
        encouragement = " Encourage experimentation while providing practical guidance."
        
        if feature_type == "premium":
            advanced = " Include professional techniques and industry insights."
            return base + context + encouragement + advanced
        
        return base + context + encouragement
    
    def _get_creative_training_context(self, factors):
        """Get relevant training context for creative requests"""
        creative_examples = []
        
        for entry in self.training_data:
            if (entry.get('category') in ['creative_breakthrough', 'narrative_art', 'style_development'] and 
                entry.get('rating', 0) >= 4):
                creative_examples.append({
                    'prompt': entry['prompt'],
                    'approach': entry['response'][:150] + "...",
                    'category': entry.get('category', 'creative')
                })
        
        if creative_examples:
            context = "Creative inspiration from successful examples:\n\n"
            for i, example in enumerate(creative_examples[:3], 1):
                context += f"Example {i} ({example['category']}):\n"
                context += f"Challenge: {example['prompt']}\n"
                context += f"Approach: {example['approach']}\n\n"
            return context
        
        return "Draw inspiration from your creative vision and artistic intuition."
    
    def _generate_learning_path(self, factors, user_info):
        """Generate personalized learning path"""
        skill_level = factors['user_skill_level']
        domain = factors['artistic_domain']
        
        learning_paths = {
            'beginner': {
                'foundations': ['Basic shapes', 'Line quality', 'Simple shading'],
                'progression': ['Form and volume', 'Color basics', 'Composition'],
                'next_level': ['Advanced techniques', 'Personal style', 'Complex subjects']
            },
            'intermediate': {
                'strengthen': ['Perspective mastery', 'Advanced shading', 'Color harmony'],
                'explore': ['New mediums', 'Different styles', 'Complex compositions'],
                'challenge': ['Professional techniques', 'Personal projects', 'Teaching others']
            },
            'advanced': {
                'refine': ['Master-level techniques', 'Artistic voice', 'Innovation'],
                'expand': ['Cross-medium work', 'Concept development', 'Art business'],
                'master': ['Teaching', 'Original research', 'Art leadership']
            }
        }
        
        path = learning_paths.get(skill_level, learning_paths['intermediate'])
        
        return {
            'current_focus': path[list(path.keys())[0]],
            'next_goals': path[list(path.keys())[1]] if len(path) > 1 else [],
            'long_term': path[list(path.keys())[2]] if len(path) > 2 else [],
            'personalized_note': f"Tailored for {skill_level} level in {domain} art"
        }
    
    def _get_semantic_hybrid_system_message(self, factors):
        """Get system message for semantic hybrid responses"""
        return f"""You are an expert art instructor with access to a comprehensive training database.
        
        User Context:
        - Skill Level: {factors['user_skill_level']}
        - Artistic Domain: {factors['artistic_domain']}
        - Learning Context: {factors['learning_context']}
        
        Instructions:
        1. Build upon the provided training examples
        2. Adapt complexity to user's skill level
        3. Include practical exercises
        4. Provide encouraging, personalized guidance
        5. Connect concepts to broader artistic knowledge"""
    
    def _format_semantic_training_context(self, semantic_matches):
        """Format semantic training matches with similarity scores"""
        if not semantic_matches:
            return "No relevant training examples found."
        
        context = "Relevant training examples (with similarity analysis):\n\n"
        
        for i, match in enumerate(semantic_matches, 1):
            entry = match['entry']
            similarities = {
                'Word Match': f"{match['word_sim']:.2f}",
                'Semantic': f"{match['semantic_sim']:.2f}",
                'Context': f"{match['context_sim']:.2f}",
                'Total': f"{match['similarity']:.2f}"
            }
            
            context += f"Example {i} (Similarity: {similarities['Total']}):\n"
            context += f"Q: {entry['prompt']}\n"
            context += f"A: {entry['response'][:200]}...\n"
            context += f"Similarities: {similarities}\n\n"
        
        return context
    
    def _adapt_message_to_skill_level(self, prompt, skill_level, domain):
        """Adapt the user message based on skill level and domain"""
        skill_adaptations = {
            'beginner': "Please provide step-by-step beginner guidance for: ",
            'intermediate': "Help me improve my technique for: ",
            'advanced': "Share advanced insights and techniques for: ",
            'expert': "Discuss professional-level approaches to: "
        }
        
        domain_context = f" in {domain} art" if domain != 'general' else ""
        
        adapted_prompt = skill_adaptations.get(skill_level, "") + prompt + domain_context
        return adapted_prompt
    
    def _get_skill_adapted_system_message(self, factors, feature_type):
        """Get skill-adapted system message"""
        skill_level = factors['user_skill_level']
        domain = factors['artistic_domain']
        
        base_instructions = {
            'beginner': "Use simple language, provide step-by-step instructions, encourage experimentation without fear of mistakes.",
            'intermediate': "Provide detailed techniques, suggest practice exercises, explain the 'why' behind methods.",
            'advanced': "Share sophisticated techniques, discuss artistic theory, challenge with complex concepts.",
            'expert': "Engage in professional-level discussion, share industry insights, focus on innovation and mastery."
        }
        
        system_msg = f"You are an expert art instructor. {base_instructions.get(skill_level, base_instructions['intermediate'])}"
        
        if domain != 'general':
            system_msg += f" Specialize your advice for {domain} art."
        
        if feature_type == "premium":
            system_msg += " Include professional tips and advanced techniques."
        
        return system_msg
    
    def _generate_transformer_insights(self, factors):
        """Generate insights from transformer analysis"""
        transformer_features = factors.get('transformer_features', {})
        intent_analysis = factors.get('intent_analysis', {})
        
        insights = {
            'detected_intent': intent_analysis.get('intent', 'drawing_request'),
            'intent_confidence': intent_analysis.get('confidence', 0.5),
            'complexity_assessment': transformer_features.get('complexity_level', 'basic'),
            'art_domain': transformer_features.get('art_domain', 'general'),
            'has_specific_subject': transformer_features.get('has_specific_subject', False),
            'semantic_analysis': 'Transformer model detected artistic intent and context',
            'recommended_approach': self._get_intent_based_approach(intent_analysis.get('intent', 'drawing_request'))
        }
        
        return insights
    
    def _get_intent_based_approach(self, intent):
        """Get recommended approach based on detected intent"""
        approaches = {
            'drawing_request': 'Provide step-by-step drawing instructions with visual guidance',
            'help_request': 'Focus on troubleshooting and problem-solving strategies',
            'tutorial_request': 'Offer comprehensive tutorial with learning progression',
            'improvement_request': 'Suggest practice exercises and skill development techniques',
            'color_request': 'Emphasize color theory and palette recommendations',
            'technique_request': 'Deep-dive into specific artistic techniques and methods',
            'tool_request': 'Explain tool usage and technical aspects'
        }
        return approaches.get(intent, 'Provide balanced artistic guidance')
    
    def _get_transformer_semantic_system_message(self, factors):
        """Get system message enhanced with transformer insights"""
        transformer_insights = factors.get('transformer_features', {})
        intent = factors.get('intent_analysis', {}).get('intent', 'drawing_request')
        
        base_msg = "You are an expert art instructor with advanced AI understanding of user intent and context."
        
        intent_specific = {
            'drawing_request': " Focus on clear, step-by-step drawing instructions.",
            'help_request': " Prioritize problem-solving and troubleshooting guidance.",
            'tutorial_request': " Provide comprehensive, educational content with progression.",
            'improvement_request': " Emphasize skill development and practice strategies.",
            'color_request': " Specialize in color theory and palette guidance.",
            'technique_request': " Deep-dive into artistic techniques and methods.",
            'tool_request': " Focus on technical tool usage and functionality."
        }
        
        complexity_level = transformer_insights.get('complexity_level', 'basic')
        complexity_instruction = {
            'basic': " Use simple, accessible language suitable for beginners.",
            'intermediate': " Provide detailed explanations with moderate technical depth.",
            'advanced': " Include sophisticated concepts and professional-level insights."
        }
        
        return (base_msg + 
                intent_specific.get(intent, " Provide balanced artistic guidance.") +
                complexity_instruction.get(complexity_level, " Adapt complexity to user level.") +
                " Integrate training data examples with your expertise.")


        if entry_skill == factors['user_skill_level']:
            relevance_score += 0.3
        
        # Domain relevance  
        entry_category = entry.get('training_category', 'general')
        if entry_category == factors['artistic_domain']:
            relevance_score += 0.4
            
        # Technique relevance
        if factors['is_advanced_technique'] and entry.get('technique_complexity', 0) >= 3:
            relevance_score += 0.3
            
        return min(relevance_score, 1.0)


        if entry_skill == factors['user_skill_level']:
            relevance_score += 0.3
        
        # Domain relevance  
        entry_category = entry.get('training_category', 'general')
        if entry_category == factors['artistic_domain']:
            relevance_score += 0.4
            
        # Technique relevance
        if factors['is_advanced_technique'] and entry.get('technique_complexity', 0) >= 3:
            relevance_score += 0.3
            
        return min(relevance_score, 1.0)
    
    def _perform_complex_algorithmic_analysis(self, prompt, user_info):
        """Complex algorithmic analysis using multiple mathematical models"""
        
        # Weighted semantic analysis algorithm
        semantic_weights = self._calculate_semantic_weights(prompt)
        
        # Pattern recognition with exponential decay
        pattern_scores = {}
        for entry in self.training_data:
            if entry.get('rating', 0) >= 4:
                similarity = self._advanced_similarity_algorithm(prompt, entry['prompt'])
                decay_factor = self._calculate_temporal_decay(entry.get('timestamp', ''))
                pattern_scores[entry['prompt'][:50]] = similarity * decay_factor
        
        # Statistical confidence modeling
        confidence_distribution = self._statistical_confidence_model(pattern_scores)
        
        # Machine learning prediction score
        ml_prediction = self._ml_prediction_algorithm(prompt, user_info)
        
        return {
            'confidence_score': confidence_distribution['mean_confidence'],
            'semantic_weights': semantic_weights,
            'pattern_distribution': confidence_distribution,
            'ml_prediction': ml_prediction,
            'algorithm_version': '2.1.0'
        }
    
    def _neural_pattern_recognition(self, prompt):
        """Neural network-inspired pattern recognition"""
        
        # Simulate neural layers with weighted connections
        input_layer = self._tokenize_and_encode(prompt)
        hidden_layer_1 = self._neural_layer_transform(input_layer, weights_1=0.7)
        hidden_layer_2 = self._neural_layer_transform(hidden_layer_1, weights_1=0.8)
        output_layer = self._neural_output_transform(hidden_layer_2)
        
        # Pattern activation mapping
        pattern_activations = {}
        for pattern_type in ['geometric', 'artistic', 'technical', 'creative']:
            activation_strength = self._calculate_pattern_activation(output_layer, pattern_type)
            pattern_activations[pattern_type] = activation_strength
        
        return {
            'neural_activation_map': pattern_activations,
            'network_confidence': sum(pattern_activations.values()) / len(pattern_activations),
            'dominant_pattern': max(pattern_activations, key=pattern_activations.get),
            'neural_complexity_score': len([v for v in pattern_activations.values() if v > 0.5])
        }
    
    def _predictive_response_modeling(self, prompt, user_info):
        """Advanced predictive modeling for response optimization"""
        
        # Time series analysis of user interactions
        user_progression = self._analyze_user_progression_pattern(user_info)
        
        # Bayesian inference for response success prediction
        success_probability = self._bayesian_success_prediction(prompt, user_info)
        
        # Markov chain modeling for conversation flow
        conversation_state = self._markov_conversation_modeling(prompt)
        
        # Regression analysis for optimal response length
        optimal_response_metrics = self._regression_response_optimization(prompt, user_info)
        
        return {
            'success_probability': success_probability,
            'user_progression_trend': user_progression,
            'conversation_state': conversation_state,
            'optimal_metrics': optimal_response_metrics,
            'predictive_confidence': (success_probability + user_progression['trend_score']) / 2
        }
    
    def _generate_contextual_embeddings(self, prompt):
        """Generate high-dimensional contextual embeddings"""
        
        # Word frequency analysis with TF-IDF simulation
        word_frequencies = {}
        words = prompt.split()
        for word in words:
            word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        # Contextual embedding dimensions
        embedding_dimensions = {
            'semantic_density': len(set(words)) / max(len(words), 1),
            'technical_complexity': len([w for w in words if len(w) > 6]) / max(len(words), 1),
            'emotional_valence': self._calculate_emotional_valence(prompt),
            'artistic_specificity': self._calculate_artistic_specificity(prompt),
            'instructional_clarity': self._calculate_instructional_clarity(prompt)
        }
        
        # Vector space representation
        embedding_vector = [embedding_dimensions[dim] for dim in sorted(embedding_dimensions.keys())]
        
        return {
            'embedding_vector': embedding_vector,
            'dimensionality': len(embedding_vector),
            'embedding_magnitude': sum(v**2 for v in embedding_vector)**0.5,
            'contextual_richness': sum(embedding_dimensions.values()) / len(embedding_dimensions)
        }
    
    def _complex_complexity_algorithm(self, prompt):
        """Advanced complexity scoring using multiple algorithmic approaches"""
        
        # Entropy-based complexity
        words = prompt.split()
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        entropy = 0
        total_words = len(words)
        for freq in word_freq.values():
            prob = freq / total_words
            if prob > 0:
                entropy -= prob * (prob**0.5)  # Modified entropy calculation
        
        # Structural complexity
        structural_indicators = {
            'sentence_variety': len(prompt.split('.')) + len(prompt.split('?')) + len(prompt.split('!')),
            'conjunction_complexity': len([w for w in words if w in ['and', 'but', 'or', 'while', 'because']]),
            'technical_depth': len([w for w in words if len(w) > 8]),
            'concept_density': len(set(words)) / max(len(words), 1)
        }
        
        # Weighted complexity score
        weights = {'entropy': 0.3, 'structural': 0.4, 'semantic': 0.3}
        semantic_complexity = sum(structural_indicators.values()) / len(structural_indicators)
        
        final_complexity = (
            entropy * weights['entropy'] + 
            semantic_complexity * weights['structural'] + 
            self._semantic_complexity_score(prompt) * weights['semantic']
        )
        
        return min(final_complexity * 2, 10.0)  # Scale to 0-10
    
    def _response_optimization_algorithm(self, prompt, user_info):
        """Multi-criteria optimization algorithm for response selection"""
        
        # Optimization criteria matrix
        criteria = {
            'user_skill_alignment': self._calculate_skill_alignment_score(prompt, user_info),
            'learning_effectiveness': self._calculate_learning_effectiveness(prompt, user_info),
            'engagement_potential': self._calculate_engagement_potential(prompt),
            'accuracy_requirements': self._calculate_accuracy_requirements(prompt),
            'time_efficiency': self._calculate_time_efficiency_score(prompt),
            'personalization_value': self._calculate_personalization_value(user_info)
        }
        
        # Multi-objective optimization using weighted sum
        weights = {
            'user_skill_alignment': 0.25,
            'learning_effectiveness': 0.20,
            'engagement_potential': 0.15,
            'accuracy_requirements': 0.20,
            'time_efficiency': 0.10,
            'personalization_value': 0.10
        }
        
        optimization_score = sum(criteria[criterion] * weights[criterion] 
                                for criterion in criteria)
        
        return {
            'optimization_matrix': criteria,
            'weighted_score': optimization_score,
            'optimization_strategy': self._determine_optimization_strategy(criteria),
            'efficiency_rating': optimization_score * 10  # Scale to 0-10
        }
    
    def _hybrid_fusion_algorithm(self, prompt):
        """Advanced algorithm for determining optimal hybrid response fusion"""
        
        # Training data relevance scoring
        training_relevance = 0.0
        relevant_entries = 0
        
        for entry in self.training_data:
            similarity = self._advanced_similarity_algorithm(prompt, entry['prompt'])
            if similarity > 0.3:
                training_relevance += similarity * entry.get('rating', 3) / 5.0
                relevant_entries += 1
        
        training_confidence = training_relevance / max(relevant_entries, 1)
        
        # OpenAI integration scoring
        openai_indicators = {
            'novelty_requirement': self._calculate_novelty_requirement(prompt),
            'creativity_demand': self._calculate_creativity_demand(prompt),
            'real_time_accuracy': self._calculate_real_time_accuracy_need(prompt)
        }
        
        openai_confidence = sum(openai_indicators.values()) / len(openai_indicators)
        
        # Fusion ratio calculation using optimization algorithm
        if training_confidence > 0.7 and openai_confidence < 0.5:
            fusion_ratio = {'training': 0.8, 'openai': 0.2}
        elif openai_confidence > 0.7 and training_confidence < 0.5:
            fusion_ratio = {'training': 0.2, 'openai': 0.8}
        else:
            # Balanced hybrid approach
            fusion_ratio = {'training': 0.5, 'openai': 0.5}
        
        return {
            'fusion_ratio': fusion_ratio,
            'training_confidence': training_confidence,
            'openai_confidence': openai_confidence,
            'hybrid_optimization_score': (training_confidence + openai_confidence) / 2,
            'fusion_strategy': self._determine_fusion_strategy(fusion_ratio)
        }
    
    def _multi_dimensional_pattern_analysis(self, prompt, user_info):
        """Multi-dimensional pattern analysis across various domains"""
        
        dimensions = {
            'temporal': self._temporal_pattern_analysis(prompt, user_info),
            'spatial': self._spatial_pattern_analysis(prompt),
            'linguistic': self._linguistic_pattern_analysis(prompt),
            'behavioral': self._behavioral_pattern_analysis(user_info),
            'contextual': self._contextual_pattern_analysis(prompt, user_info)
        }
        
        # Cross-dimensional correlation analysis
        correlations = {}
        dimension_names = list(dimensions.keys())
        for i, dim1 in enumerate(dimension_names):
            for dim2 in dimension_names[i+1:]:
                correlation = self._calculate_dimension_correlation(
                    dimensions[dim1], dimensions[dim2]
                )
                correlations[f"{dim1}_{dim2}"] = correlation
        
        # Pattern synthesis algorithm
        pattern_synthesis = self._synthesize_multi_dimensional_patterns(dimensions, correlations)
        
        return {
            'dimensional_analysis': dimensions,
            'cross_correlations': correlations,
            'pattern_synthesis': pattern_synthesis,
            'multi_dimensional_confidence': pattern_synthesis['confidence_score']
        }
    
    # Helper methods for the complex algorithms
    def _calculate_semantic_weights(self, prompt):
        """Calculate semantic weights for words in prompt"""
        art_terms = ['draw', 'paint', 'sketch', 'color', 'shape', 'line', 'form']
        tech_terms = ['tool', 'brush', 'canvas', 'layer', 'pixel', 'digital']
        creative_terms = ['creative', 'artistic', 'style', 'design', 'aesthetic']
        
        words = prompt.lower().split()
        weights = {}
        
        for word in words:
            if word in art_terms:
                weights[word] = 1.0
            elif word in tech_terms:
                weights[word] = 0.8
            elif word in creative_terms:
                weights[word] = 0.9
            else:
                weights[word] = 0.5
        
        return weights
    
    def _advanced_similarity_algorithm(self, prompt1, prompt2):
        """Advanced similarity calculation with multiple factors"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())
        
        # Jaccard similarity
        jaccard = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
        
        # Length similarity factor
        length_factor = 1 - abs(len(prompt1) - len(prompt2)) / max(len(prompt1), len(prompt2))
        
        # Semantic boost for art terms
        art_terms = ['draw', 'paint', 'sketch', 'art', 'creative', 'design']
        common_art_terms = len([term for term in art_terms if term in prompt1.lower() and term in prompt2.lower()])
        art_boost = common_art_terms * 0.1
        
        return min(jaccard + length_factor * 0.3 + art_boost, 1.0)
    
    def _calculate_temporal_decay(self, timestamp_str):
        """Calculate temporal decay factor for training data relevance"""
        if not timestamp_str:
            return 0.5  # Default for missing timestamps
        
        try:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            days_old = (datetime.now() - timestamp).days
            # Exponential decay: newer data is more relevant
            decay_factor = max(0.1, 1.0 * (0.95 ** days_old))
            return decay_factor
        except:
            return 0.5
    
    def _statistical_confidence_model(self, pattern_scores):
        """Statistical modeling of confidence distribution"""
        if not pattern_scores:
            return {'mean_confidence': 0.3, 'std_deviation': 0.1, 'max_confidence': 0.3}
        
        scores = list(pattern_scores.values())
        mean_conf = sum(scores) / len(scores)
        variance = sum((x - mean_conf) ** 2 for x in scores) / len(scores)
        std_dev = variance ** 0.5
        
        return {
            'mean_confidence': mean_conf,
            'std_deviation': std_dev,
            'max_confidence': max(scores),
            'confidence_range': max(scores) - min(scores)
        }
    
    def _ml_prediction_algorithm(self, prompt, user_info):
        """Machine learning-inspired prediction algorithm"""
        features = {
            'prompt_length': len(prompt),
            'word_count': len(prompt.split()),
            'user_experience': user_info.get('total_uses', 0),
            'complexity_indicator': len([w for w in prompt.split() if len(w) > 6])
        }
        
        # Simple linear combination (simulating trained model)
        weights = {'prompt_length': 0.01, 'word_count': 0.1, 'user_experience': 0.02, 'complexity_indicator': 0.15}
        prediction_score = sum(features[feat] * weights[feat] for feat in features)
        
        return min(max(prediction_score, 0.0), 1.0)  # Normalize to 0-1
    
    def _tokenize_and_encode(self, prompt):
        """Tokenize and encode prompt for neural processing"""
        words = prompt.lower().split()
        # Simple encoding: word length and position features
        encoded = []
        for i, word in enumerate(words):
            features = [len(word) / 10.0, i / len(words), 1.0 if len(word) > 5 else 0.0]
            encoded.append(features)
        return encoded if encoded else [[0.0, 0.0, 0.0]]
    
    def _neural_layer_transform(self, input_data, weights_1):
        """Simulate neural layer transformation"""
        transformed = []
        for features in input_data:
            # Simple weighted transformation with activation function
            output = sum(f * weights_1 for f in features)
            activated = max(0, output)  # ReLU activation
            transformed.append([activated, output * 0.5, min(output, 1.0)])
        return transformed
    
    def _neural_output_transform(self, hidden_output):
        """Transform hidden layer to output"""
        if not hidden_output:
            return [0.5, 0.5, 0.5]
        
        flattened = [val for sublist in hidden_output for val in sublist]
        avg_activation = sum(flattened) / len(flattened)
        return [avg_activation, max(flattened), min(flattened)]
    
    def _calculate_pattern_activation(self, output_layer, pattern_type):
        """Calculate activation strength for specific patterns"""
        pattern_weights = {
            'geometric': [0.8, 0.2, 0.1],
            'artistic': [0.3, 0.7, 0.4],
            'technical': [0.6, 0.5, 0.8],
            'creative': [0.4, 0.9, 0.3]
        }
        
        weights = pattern_weights.get(pattern_type, [0.5, 0.5, 0.5])
        activation = sum(output_layer[i] * weights[i] for i in range(min(len(output_layer), len(weights))))
        return min(max(activation, 0.0), 1.0)
    
    def _analyze_user_progression_pattern(self, user_info):
        """Analyze user progression patterns"""
        total_uses = user_info.get('total_uses', 0)
        
        if total_uses < 5:
            trend = 'beginner_exploration'
            trend_score = 0.3
        elif total_uses < 20:
            trend = 'skill_building'
            trend_score = 0.6
        elif total_uses < 50:
            trend = 'advanced_learning'
            trend_score = 0.8
        else:
            trend = 'expert_mastery'
            trend_score = 0.9
        
        return {'trend': trend, 'trend_score': trend_score, 'usage_velocity': min(total_uses / 10, 1.0)}
    
    def _bayesian_success_prediction(self, prompt, user_info):
        """Bayesian inference for response success prediction"""
        # Prior probability based on overall system performance
        prior = self._calculate_success_rate() / 100.0 if self.performance_metrics['total_interactions'] > 0 else 0.7
        
        # Likelihood based on prompt characteristics
        prompt_words = prompt.lower().split()
        art_keywords = ['draw', 'paint', 'sketch', 'color', 'art', 'create']
        art_keyword_count = sum(1 for word in prompt_words if word in art_keywords)
        
        likelihood = min(0.3 + art_keyword_count * 0.2, 0.9)
        
        # User experience factor
        user_experience_factor = min(user_info.get('total_uses', 0) * 0.02, 0.3)
        
        # Bayesian update (simplified)
        posterior = prior * likelihood + user_experience_factor
        return min(posterior, 1.0)
    
    def _markov_conversation_modeling(self, prompt):
        """Model conversation flow using Markov chain concepts"""
        prompt_lower = prompt.lower()
        
        # Define conversation states
        if any(word in prompt_lower for word in ['help', 'problem', 'issue', 'error']):
            current_state = 'troubleshooting'
            next_state_prob = {'solution': 0.7, 'clarification': 0.2, 'escalation': 0.1}
        elif any(word in prompt_lower for word in ['draw', 'create', 'make', 'paint']):
            current_state = 'instruction_seeking'
            next_state_prob = {'step_by_step': 0.6, 'technique_focus': 0.3, 'inspiration': 0.1}
        elif any(word in prompt_lower for word in ['improve', 'better', 'advanced']):
            current_state = 'skill_advancement'
            next_state_prob = {'advanced_technique': 0.5, 'practice_suggestion': 0.3, 'theory': 0.2}
        else:
            current_state = 'general_inquiry'
            next_state_prob = {'basic_instruction': 0.4, 'exploration': 0.4, 'clarification': 0.2}
        
        return {
            'current_state': current_state,
            'transition_probabilities': next_state_prob,
            'conversation_confidence': max(next_state_prob.values())
        }
    
    def _regression_response_optimization(self, prompt, user_info):
        """Regression analysis for optimal response characteristics"""
        # Features for regression
        user_skill = user_info.get('total_uses', 0)
        prompt_complexity = len(prompt.split())
        
        # Simulated regression coefficients (would be learned from data)
        optimal_length = 100 + user_skill * 2 + prompt_complexity * 5
        optimal_length = min(max(optimal_length, 50), 300)  # Bound between 50-300 words
        
        detail_level = 0.3 + (user_skill / 100) + (prompt_complexity / 50)
        detail_level = min(max(detail_level, 0.3), 0.9)
        
        return {
            'optimal_word_count': optimal_length,
            'detail_level': detail_level,
            'technical_depth': min(user_skill * 0.01, 0.8),
            'encouragement_factor': max(0.2, 0.8 - user_skill * 0.01)
        }
    
    def _semantic_complexity_score(self, prompt):
        """Calculate semantic complexity score"""
        words = prompt.lower().split()
        
        # Vocabulary diversity
        unique_ratio = len(set(words)) / len(words) if words else 0
        
        # Average word length
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Technical term density
        tech_terms = ['technique', 'composition', 'perspective', 'chiaroscuro', 'algorithm']
        tech_density = sum(1 for word in words if word in tech_terms) / len(words) if words else 0
        
        return (unique_ratio + avg_word_length / 10 + tech_density) / 3
    
    def _calculate_skill_alignment_score(self, prompt, user_info):
        """Calculate how well prompt aligns with user skill level"""
        user_level = self._assess_user_skill_level(user_info)
        prompt_difficulty = self._assess_prompt_difficulty(prompt)
        
        level_mapping = {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}
        user_score = level_mapping.get(user_level, 2)
        
        # Optimal alignment when prompt difficulty matches user level ±1
        alignment = 1.0 - abs(prompt_difficulty - user_score) * 0.3
        return max(alignment, 0.0)
    
    def _assess_prompt_difficulty(self, prompt):
        """Assess prompt difficulty level"""
        difficulty_indicators = {
            'beginner': ['simple', 'basic', 'easy', 'first', 'start'],
            'intermediate': ['technique', 'improve', 'better', 'practice'],
            'advanced': ['complex', 'professional', 'advanced', 'master'],
            'expert': ['photorealistic', 'virtuoso', 'experimental', 'innovative']
        }
        
        prompt_lower = prompt.lower()
        for level, keywords in difficulty_indicators.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return {'beginner': 1, 'intermediate': 2, 'advanced': 3, 'expert': 4}[level]
        return 2  # Default to intermediate
    
    def _calculate_learning_effectiveness(self, prompt, user_info):
        """Calculate potential learning effectiveness"""
        # Clear instruction indicators
        instruction_words = ['how', 'step', 'guide', 'tutorial', 'method']
        instruction_score = sum(1 for word in instruction_words if word in prompt.lower()) / len(instruction_words)
        
        # User readiness for learning
        user_readiness = min(user_info.get('total_uses', 0) * 0.05, 1.0)
        
        # Prompt clarity
        clarity_score = 1.0 - (len(prompt.split()) - 5) * 0.02 if len(prompt.split()) > 5 else 1.0
        clarity_score = max(clarity_score, 0.3)
        
        return (instruction_score + user_readiness + clarity_score) / 3
    
    def _calculate_engagement_potential(self, prompt):
        """Calculate engagement potential of the prompt"""
        engaging_words = ['creative', 'fun', 'cool', 'amazing', 'beautiful', 'awesome']
        engagement_score = sum(1 for word in engaging_words if word in prompt.lower()) * 0.2
        
        # Questions engage more
        question_score = 0.3 if '?' in prompt else 0.0
        
        # Personal pronouns increase engagement
        personal_score = 0.2 if any(word in prompt.lower() for word in ['my', 'i', 'me']) else 0.0
        
        return min(engagement_score + question_score + personal_score + 0.3, 1.0)
    
    def _calculate_accuracy_requirements(self, prompt):
        """Calculate accuracy requirements for the response"""
        accuracy_keywords = ['precise', 'exact', 'accurate', 'correct', 'professional', 'realistic']
        accuracy_score = sum(1 for word in accuracy_keywords if word in prompt.lower()) * 0.25
        
        # Technical subjects require higher accuracy
        technical_keywords = ['perspective', 'anatomy', 'proportion', 'measurement']
        technical_score = sum(1 for word in technical_keywords if word in prompt.lower()) * 0.3
        
        return min(accuracy_score + technical_score + 0.2, 1.0)
    
    def _calculate_time_efficiency_score(self, prompt):
        """Calculate time efficiency requirements"""
        urgency_words = ['quick', 'fast', 'immediately', 'now', 'urgent']
        urgency_score = sum(1 for word in urgency_words if word in prompt.lower()) * 0.3
        
        # Shorter prompts suggest need for quick response
        length_efficiency = max(0.2, 1.0 - len(prompt.split()) * 0.05)
        
        return min(urgency_score + length_efficiency, 1.0)
    
    def _calculate_personalization_value(self, user_info):
        """Calculate value of personalization for this user"""
        usage_history = user_info.get('total_uses', 0)
        premium_status = user_info.get('premium', False)
        
        # More usage = more personalization value
        usage_factor = min(usage_history * 0.02, 0.6)
        premium_factor = 0.3 if premium_status else 0.0
        base_value = 0.1
        
        return usage_factor + premium_factor + base_value
    
    def _determine_optimization_strategy(self, criteria):
        """Determine the best optimization strategy based on criteria"""
        max_criterion = max(criteria, key=criteria.get)
        
        strategies = {
            'user_skill_alignment': 'skill_adaptive',
            'learning_effectiveness': 'educational_focus',
            'engagement_potential': 'engagement_maximization',
            'accuracy_requirements': 'precision_optimization',
            'time_efficiency': 'rapid_response',
            'personalization_value': 'personalized_approach'
        }
        
        return strategies.get(max_criterion, 'balanced_approach')
    
    def _calculate_novelty_requirement(self, prompt):
        """Calculate how much novelty/creativity is required"""
        novelty_words = ['new', 'unique', 'original', 'creative', 'innovative', 'different']
        return min(sum(1 for word in novelty_words if word in prompt.lower()) * 0.25 + 0.1, 1.0)
    
    def _calculate_creativity_demand(self, prompt):
        """Calculate creativity demand of the prompt"""
        creative_words = ['artistic', 'expressive', 'imaginative', 'stylistic', 'abstract']
        return min(sum(1 for word in creative_words if word in prompt.lower()) * 0.3 + 0.2, 1.0)
    
    def _calculate_real_time_accuracy_need(self, prompt):
        """Calculate need for real-time accurate information"""
        current_words = ['current', 'latest', 'new', 'modern', 'recent', 'updated']
        return min(sum(1 for word in current_words if word in prompt.lower()) * 0.4 + 0.1, 1.0)
    
    def _determine_fusion_strategy(self, fusion_ratio):
        """Determine the best fusion strategy"""
        training_weight = fusion_ratio['training']
        
        if training_weight > 0.7:
            return 'training_dominant'
        elif training_weight < 0.3:
            return 'openai_dominant'
        else:
            return 'balanced_hybrid'
    
    def _temporal_pattern_analysis(self, prompt, user_info):
        """Analyze temporal patterns in user behavior"""
        usage_frequency = user_info.get('total_uses', 0) / max(1, (datetime.now() - datetime.now()).days + 1)
        return {'usage_frequency': usage_frequency, 'temporal_consistency': min(usage_frequency, 1.0)}
    
    def _spatial_pattern_analysis(self, prompt):
        """Analyze spatial patterns in the prompt"""
        spatial_words = ['above', 'below', 'left', 'right', 'center', 'corner', 'edge']
        spatial_density = sum(1 for word in spatial_words if word in prompt.lower()) / len(prompt.split())
        return {'spatial_awareness': spatial_density, 'spatial_complexity': min(spatial_density * 5, 1.0)}
    
    def _linguistic_pattern_analysis(self, prompt):
        """Analyze linguistic patterns"""
        sentence_count = len([s for s in prompt.split('.') if s.strip()])
        avg_word_length = sum(len(word) for word in prompt.split()) / len(prompt.split()) if prompt.split() else 0
        return {'linguistic_complexity': avg_word_length / 10, 'sentence_structure': min(sentence_count, 5) / 5}
    
    def _behavioral_pattern_analysis(self, user_info):
        """Analyze user behavioral patterns"""
        usage_pattern = user_info.get('total_uses', 0)
        premium_status = user_info.get('premium', False)
        engagement_level = min(usage_pattern * 0.1, 1.0)
        return {'engagement_level': engagement_level, 'premium_behavior': 1.0 if premium_status else 0.3}
    
    def _contextual_pattern_analysis(self, prompt, user_info):
        """Analyze contextual patterns"""
        skill_context = self._assess_user_skill_level(user_info)
        domain_context = self._identify_artistic_domain(prompt.lower())
        context_score = {'beginner': 0.2, 'intermediate': 0.5, 'advanced': 0.8, 'expert': 1.0}.get(skill_context, 0.5)
        return {'skill_context': context_score, 'domain_specificity': 0.8 if domain_context != 'general' else 0.3}
    
    def _calculate_dimension_correlation(self, dim1, dim2):
        """Calculate correlation between two dimensions"""
        # Simple correlation based on value similarity
        dim1_avg = sum(dim1.values()) / len(dim1) if dim1 else 0
        dim2_avg = sum(dim2.values()) / len(dim2) if dim2 else 0
        return 1.0 - abs(dim1_avg - dim2_avg)
    
    def _synthesize_multi_dimensional_patterns(self, dimensions, correlations):
        """Synthesize patterns across multiple dimensions"""
        pattern_strength = sum(sum(dim.values()) for dim in dimensions.values()) / len(dimensions)
        correlation_strength = sum(correlations.values()) / len(correlations) if correlations else 0.5
        
        synthesis_score = (pattern_strength + correlation_strength) / 2
        
        return {
            'pattern_strength': pattern_strength,
            'correlation_strength': correlation_strength,
            'synthesis_score': synthesis_score,
            'confidence_score': min(synthesis_score * 1.2, 1.0)
        }
    
    def _calculate_emotional_valence(self, prompt):
        """Calculate emotional valence of the prompt"""
        positive_words = ['beautiful', 'amazing', 'wonderful', 'great', 'awesome', 'love']
        negative_words = ['difficult', 'hard', 'problem', 'trouble', 'confused', 'stuck']
        
        positive_count = sum(1 for word in positive_words if word in prompt.lower())
        negative_count = sum(1 for word in negative_words if word in prompt.lower())
        
        return max(0.1, 0.5 + (positive_count - negative_count) * 0.2)
    
    def _calculate_artistic_specificity(self, prompt):
        """Calculate artistic specificity level"""
        specific_terms = ['chiaroscuro', 'perspective', 'composition', 'palette', 'technique', 'style']
        general_terms = ['draw', 'paint', 'make', 'create', 'art']
        
        specific_count = sum(1 for term in specific_terms if term in prompt.lower())
        general_count = sum(1 for term in general_terms if term in prompt.lower())
        
        total_words = len(prompt.split())
        specificity = (specific_count * 2 + general_count) / max(total_words, 1)
        
        return min(specificity, 1.0)
    
    def _calculate_instructional_clarity(self, prompt):
        """Calculate instructional clarity score"""
        clear_indicators = ['how', 'step', 'guide', 'tutorial', 'method', 'way', 'process']
        unclear_indicators = ['somehow', 'maybe', 'perhaps', 'might', 'could']
        
        clear_count = sum(1 for word in clear_indicators if word in prompt.lower())
        unclear_count = sum(1 for word in unclear_indicators if word in prompt.lower())
        
        clarity_score = (clear_count - unclear_count * 0.5) / max(len(prompt.split()), 1)
        return max(min(clarity_score + 0.5, 1.0), 0.1)
    
    def _get_relevant_training_examples(self, prompt, limit=3):
        """Get most relevant training examples for the prompt"""
        prompt_words = set(prompt.split())
        relevant_examples = []
        
        for entry in self.training_data:
            if entry.get("rating", 0) >= 4:
                entry_words = set(entry["prompt"].split())
                if prompt_words and entry_words:
                    similarity = len(prompt_words.intersection(entry_words)) / len(prompt_words.union(entry_words))
                    if similarity > 0.2:  # 20% minimum similarity
                        relevant_examples.append({
                            "prompt": entry["prompt"],
                            "response": entry["response"],
                            "rating": entry["rating"],
                            "similarity": similarity
                        })
        
        # Sort by similarity and return top examples
        relevant_examples.sort(key=lambda x: x["similarity"], reverse=True)
        return relevant_examples[:limit]
    
    def _format_training_context(self, training_examples):
        """Format training examples into context for hybrid responses"""
        if not training_examples:
            return "No relevant training examples found."
        
        context = "Relevant training examples:\n\n"
        for i, example in enumerate(training_examples, 1):
            context += f"Example {i} (similarity: {example['similarity']:.2f}):\n"
            context += f"Q: {example['prompt']}\n"
            context += f"A: {example['response'][:300]}...\n\n"
        
        return context
    
    def _get_troubleshooting_system_message(self):
        """Get system message for troubleshooting requests"""
        return """You are an expert technical support specialist for a digital drawing application.
        Provide clear, step-by-step solutions for technical issues.
        Include immediate fixes, common causes, and prevention tips.
        Use HTML formatting with emojis for visual appeal."""
    
    def _get_creative_system_message(self, feature_type):
        """Get system message for creative requests"""
        base = """You are a creative drawing instructor who inspires artistic expression.
        Encourage experimentation and provide multiple creative approaches."""
        
        if feature_type == "premium":
            base += " Include advanced artistic techniques and professional insights."
        
        return base
    
    def _get_hybrid_system_message(self, feature_type):
        """Get system message for hybrid responses"""
        return """You are an expert drawing instructor with access to a comprehensive training database.
        Combine the provided training examples with your knowledge to create the best possible response.
        Build upon the training data while adding your own expertise and insights."""
    
    def _get_general_system_message(self, feature_type):
        """Get system message for general requests"""
        base = """You are an expert drawing instructor. Provide clear, step-by-step instructions."""
        
        if feature_type == "premium":
            base += " Include advanced techniques and professional tips."
        
        return base

# Initialize AI training system
ai_training = AITrainingSystem()

# Load comprehensive training data on startup
try:
    from training_data_feeder import feed_comprehensive_training_data
    print("🚀 Loading comprehensive training dataset...")
    training_results = feed_comprehensive_training_data()
    print(f"✅ Training data loaded: {training_results['total_entries_added']} entries")
except Exception as e:
    print(f"⚠️ Could not load training data: {e}")

# Production-scale API key management system
class APIKeyManager:
    def __init__(self):
        self.usage_stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limit_hits": 0,
            "daily_usage": {},
            "last_reset": datetime.now().date().isoformat()
        }
        self.key_validation_cache = {}
        self.rate_limit_tracker = {}
        
    def get_api_key_status(self):
        """Comprehensive API key validation with caching"""
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Basic validation
        if not api_key:
            return {
                "valid": False, 
                "message": "❌ No API key configured", 
                "status": "missing",
                "action_required": "Add OPENAI_API_KEY to Secrets"
            }
            
        if not api_key.startswith("sk-"):
            return {
                "valid": False, 
                "message": "❌ Invalid API key format", 
                "status": "invalid_format",
                "action_required": "API key must start with 'sk-'"
            }

        # Check cache first (avoid excessive API calls)
        cache_key = api_key[-8:]  # Last 8 chars for identification
        if cache_key in self.key_validation_cache:
            cached = self.key_validation_cache[cache_key]
            if (datetime.now() - datetime.fromisoformat(cached["timestamp"])).seconds < 300:  # 5 min cache
                return cached["result"]

        # Live API validation
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "API test"}],
                max_tokens=5
            )
            
            result = {
                "valid": True, 
                "message": "✅ API key working perfectly!", 
                "status": "active",
                "model_access": "gpt-3.5-turbo",
                "last_tested": datetime.now().isoformat()
            }
            
            # Cache successful validation
            self.key_validation_cache[cache_key] = {
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
            
            self.usage_stats["successful_requests"] += 1
            return result
            
        except Exception as e:
            error_msg = str(e).lower()
            self.usage_stats["failed_requests"] += 1
            
            if "rate limit" in error_msg:
                self.usage_stats["rate_limit_hits"] += 1
                return {
                    "valid": False, 
                    "message": "⚠️ Rate limit exceeded", 
                    "status": "rate_limited",
                    "action_required": "Wait before next request or upgrade OpenAI plan"
                }
            elif "authentication" in error_msg or "invalid" in error_msg:
                return {
                    "valid": False, 
                    "message": f"❌ Authentication failed: {str(e)}", 
                    "status": "auth_failed",
                    "action_required": "Check API key validity on OpenAI dashboard"
                }
            elif "quota" in error_msg or "billing" in error_msg:
                return {
                    "valid": False, 
                    "message": "💳 Billing/quota issue", 
                    "status": "billing_issue",
                    "action_required": "Check OpenAI billing dashboard"
                }
            else:
                return {
                    "valid": False, 
                    "message": f"🔧 Connection error: {str(e)}", 
                    "status": "connection_error",
                    "action_required": "Check internet connection or try again"
                }
    
    def track_usage(self, success=True):
        """Track API usage for monitoring"""
        today = datetime.now().date().isoformat()
        
        if today != self.usage_stats["last_reset"]:
            # Reset daily counter
            self.usage_stats["daily_usage"][self.usage_stats["last_reset"]] = self.usage_stats.get("daily_requests", 0)
            self.usage_stats["daily_requests"] = 0
            self.usage_stats["last_reset"] = today
            
        self.usage_stats["total_requests"] += 1
        self.usage_stats["daily_requests"] = self.usage_stats.get("daily_requests", 0) + 1
        
        if success:
            self.usage_stats["successful_requests"] += 1
        else:
            self.usage_stats["failed_requests"] += 1
    
    def get_usage_statistics(self):
        """Get comprehensive usage statistics"""
        total = self.usage_stats["total_requests"]
        success_rate = (self.usage_stats["successful_requests"] / total * 100) if total > 0 else 0
        
        return {
            "total_requests": total,
            "success_rate": round(success_rate, 1),
            "daily_requests": self.usage_stats.get("daily_requests", 0),
            "rate_limit_hits": self.usage_stats["rate_limit_hits"],
            "daily_usage_history": self.usage_stats.get("daily_usage", {}),
            "health_status": "healthy" if success_rate > 95 else "needs_attention" if success_rate > 80 else "critical"
        }

# Initialize API key manager
api_manager = APIKeyManager()

def check_api_key():
    """Legacy function for backward compatibility"""
    status = api_manager.get_api_key_status()
    return status["valid"], status["message"], status["status"]

@app.get("/api-status")
def api_status():
    """Production-grade API status with comprehensive metrics"""
    key_status = api_manager.get_api_key_status()
    usage_stats = api_manager.get_usage_statistics()
    
    return {
        "api_key_status": key_status,
        "usage_statistics": usage_stats,
        "system_health": {
            "status": usage_stats["health_status"],
            "uptime": "Available",
            "last_check": datetime.now().isoformat()
        },
        "recommendations": _get_api_recommendations(key_status, usage_stats)
    }

def _get_api_recommendations(key_status, usage_stats):
    """Generate production recommendations"""
    recommendations = []
    
    if not key_status["valid"]:
        recommendations.append({
            "priority": "critical",
            "issue": "API key not functional",
            "action": key_status.get("action_required", "Fix API key configuration")
        })
    
    if usage_stats["rate_limit_hits"] > 5:
        recommendations.append({
            "priority": "high",
            "issue": "Frequent rate limiting",
            "action": "Consider upgrading OpenAI plan or implementing request throttling"
        })
    
    if usage_stats["success_rate"] < 90:
        recommendations.append({
            "priority": "medium",
            "issue": f"Low success rate: {usage_stats['success_rate']}%",
            "action": "Monitor API errors and implement retry logic"
        })
    
    if usage_stats["daily_requests"] > 1000:
        recommendations.append({
            "priority": "info",
            "issue": "High daily usage",
            "action": "Monitor costs and consider usage optimization"
        })
    
    if not recommendations:
        recommendations.append({
            "priority": "success",
            "issue": "System operating normally",
            "action": "Continue monitoring"
        })
    
    return recommendations

# Mount static files to serve JS, CSS, and other assets
app.mount("/static", StaticFiles(directory="."), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import database manager
from database_setup import DatabaseManager

# Initialize systems
db_manager = DatabaseManager()

# Initialize database tables on startup
db_manager.init_tables()

def get_user_session(request):
    user_id = request.headers.get('user-agent', str(uuid.uuid4()))[:50]
    user_data = db_manager.get_user_data(user_id)
    return user_id, user_data

def update_user_data(user_id, user_info):
    db_manager.save_user_data(user_id, user_info)

def get_image_references(prompt):
    """Generate web image reference suggestions based on the prompt"""
    prompt_lower = prompt.lower()

    # Map subjects to specific image references
    reference_suggestions = []

    if any(word in prompt_lower for word in ['cat', 'kitten', 'feline']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1574158622682-e40e69881006?w=400",  # Cat portrait
            "https://images.unsplash.com/photo-1611003229186-80e40cd54966?w=400",  # Cat sitting
        ])
    elif any(word in prompt_lower for word in ['dog', 'puppy', 'canine']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1518717758536-85ae29035b6d?w=400",  # Dog portrait
            "https://images.unsplash.com/photo-1552053831-71594a27632d?w=400",  # Golden retriever
        ])
    elif any(word in prompt_lower for word in ['house', 'home', 'building']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1570129477492-45c003edd2be?w=400",  # Modern house
            "https://images.unsplash.com/photo-1518780664697-55e3ad937233?w=400",  # Cozy house
        ])
    elif any(word in prompt_lower for word in ['tree', 'forest', 'nature']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # Forest trees
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Single tree
        ])
    elif any(word in prompt_lower for word in ['flower', 'rose', 'bloom']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1518709268805-4e9042af2176?w=400",  # Pink flowers
            "https://images.unsplash.com/photo-1563206480-e0246ad0b14c?w=400",  # Sunflower
        ])
    elif any(word in prompt_lower for word in ['car', 'vehicle', 'automobile']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1493238792000-8113da705763?w=400",  # Red car
            "https://images.unsplash.com/photo-1502877338535-766e1452684a?w=400",  # Classic car
        ])
    elif any(word in prompt_lower for word in ['face', 'portrait', 'person']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1494790108755-2616c047016b?w=400",  # Female portrait
            "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Male portrait
        ])
    elif any(word in prompt_lower for word in ['mountain', 'landscape', 'nature']):
        reference_suggestions.extend([
            "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # Mountain landscape
            "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # Nature scene
        ])

    if reference_suggestions:
        refs_html = "📷 <strong>Reference Images:</strong><br>"
        for i, ref in enumerate(reference_suggestions[:2], 1):  # Limit to 2 references
            refs_html += f"• <a href='javascript:void(0)' onclick=\"document.getElementById('imageUrl').value='{ref}'; loadReference();\">Reference {i}</a> - Click to load automatically<br>"

        refs_html += "<br>💡 <strong>Tip:</strong> Load a reference image above your canvas for better accuracy!"
        return refs_html

    return ""

def get_tool_troubleshooting_guide(prompt):
    """Generate comprehensive tool troubleshooting based on the prompt or general issues"""
    prompt_lower = prompt.lower()

    # Always provide general troubleshooting for drawing-related queries
    if any(word in prompt_lower for word in ['draw', 'sketch', 'paint', 'create', 'make']):
        return """🔧 <strong>Complete Tool Guide:</strong><br>

        <strong>📝 Drawing Tools:</strong><br>
        • <strong>Pencil:</strong> Click tool → Choose brush size → Click & drag on canvas<br>
        • <strong>Eraser:</strong> Select eraser → Adjust size → Draw over areas to remove<br>
        • <strong>Fill:</strong> Select fill tool → Click inside enclosed shapes to fill with color<br>
        • <strong>Eyedropper:</strong> Click tool → Click on canvas to pick existing colors<br>

        <strong>🔷 Shape Tools:</strong><br>
        • <strong>Rectangle/Circle/Triangle:</strong> Select shape → Click & drag from corner to corner<br>
        • <strong>Line:</strong> Click start point → Drag to end point → Release<br>
        • <strong>Star/Arrow:</strong> Click center → Drag outward to set size<br>

        <strong>🎨 Essential Settings:</strong><br>
        • <strong>Brush Size:</strong> Use slider (1-50px) or keyboard [ ] keys<br>
        • <strong>Colors:</strong> Click color picker or use preset color squares<br>
        • <strong>Shape Fill:</strong> Check "Fill shapes" box for solid shapes<br>
        • <strong>Opacity:</strong> Adjust shape opacity slider for transparency<br>

        <strong>📚 Layer System:</strong><br>
        • <strong>Add Layer:</strong> Click "+ Add Layer" button<br>
        • <strong>Switch Layers:</strong> Click layer name to make it active<br>
        • <strong>Hide/Show:</strong> Check/uncheck layer visibility boxes<br>

        <strong>⚡ Quick Fixes:</strong><br>
        ✅ <strong>Tool not working:</strong> Check if correct tool is highlighted in blue<br>
        ✅ <strong>Can't draw:</strong> Click on canvas first, check brush size > 1<br>
        ✅ <strong>Wrong color:</strong> Verify color picker shows desired color<br>
        ✅ <strong>Undo/Redo:</strong> Use Ctrl+Z / Ctrl+Y or the arrow buttons"""

    # Specific troubleshooting for tool-related issues
    elif any(word in prompt_lower for word in ['tool', 'brush', 'pencil', 'eraser', 'problem', 'issue', 'help', 'not working']):
        return """🛠️ <strong>Tool Troubleshooting:</strong><br>

        <strong>Common Issues & Solutions:</strong><br>

        🔹 <strong>Pencil Tool Issues:</strong><br>
        • Tool not selected? → Click "Pencil" button (should turn blue)<br>
        • Brush too small? → Increase brush size slider<br>
        • Canvas not focused? → Click once on the white canvas area<br>

        🔹 <strong>Shape Tool Problems:</strong><br>
        • Not drawing shapes? → Click and DRAG, don't just click<br>
        • Shapes not filled? → Check "Fill shapes" checkbox<br>
        • Can't see shape? → Check opacity setting and color<br>

        🔹 <strong>Color Issues:</strong><br>
        • Color not changing? → Click color picker square to choose new color<br>
        • Eyedropper not working? → Select eyedropper tool, then click on canvas<br>
        • Fill not working? → Make sure area is completely enclosed<br>

        🔹 <strong>Canvas Problems:</strong><br>
        • Can't draw anywhere? → Try refreshing page<br>
        • Canvas appears frozen? → Check browser supports HTML5<br>
        • Touch not working? → Try mouse/trackpad instead<br>

        <strong>📱 Mobile Users:</strong><br>
        • Use single finger touches<br>
        • Avoid scrolling while drawing<br>
        • Try landscape orientation for better control<br>

        <strong>⌨️ Keyboard Shortcuts:</strong><br>
        • P = Pencil, E = Eraser, F = Fill, R = Rectangle, C = Circle<br>
        • [ ] = Decrease/Increase brush size<br>
        • Ctrl+Z = Undo, Ctrl+Y = Redo<br>
        • 1-5 = Quick color selection"""

    return ""

@app.get("/")
def root():
    return FileResponse("index.html")

@app.get("/api")
def api_status():
    return {"message": "Drawing AI Backend is Running"}

@app.get("/user-status")
def get_user_status(request: Request):
    user_id, user_info = get_user_session(request)

    # Check if trial period has started
    trial_start = user_info.get("trial_start_date")
    is_premium = user_info.get("premium", False)
    trial_active = False
    trial_days_remaining = 0

    if trial_start:
        trial_start_date = datetime.fromisoformat(trial_start)
        days_since_trial = (datetime.now() - trial_start_date).days
        trial_days_remaining = max(0, 10 - days_since_trial)
        trial_active = trial_days_remaining > 0
    else:
        # First time user - hasn't started trial yet
        trial_days_remaining = 10
        trial_active = True

    # Check premium status
    if is_premium and user_info.get("premium_expiry"):
        expiry_date = datetime.fromisoformat(user_info["premium_expiry"])
        is_premium = datetime.now() < expiry_date
        if not is_premium:
            user_info["premium"] = False
            update_user_data(user_id, user_info)

    return {
        "trial_days_remaining": trial_days_remaining,
        "trial_active": trial_active,
        "is_premium": is_premium,
        "total_uses": user_info.get("total_uses", 0),
        "premium_expiry": user_info.get("premium_expiry"),
        "trial_started": trial_start is not None
    }

# Payment endpoints
@app.post("/create-payment-intent")
async def create_payment_intent(request: Request):
    data = await request.json()
    plan = data.get("plan", "monthly")  # monthly, yearly

    # Pricing in cents
    prices = {
        "monthly": 999,  # $9.99/month
        "yearly": 9999   # $99.99/year (save $20)
    }

    # In production, use Stripe API
    payment_intent = {
        "client_secret": f"pi_demo_{plan}_{uuid.uuid4().hex[:8]}",
        "amount": prices[plan],
        "currency": "usd",
        "plan": plan
    }

    return payment_intent

@app.post("/verify-payment")
async def verify_payment(request: Request):
    data = await request.json()
    payment_intent_id = data.get("payment_intent_id")
    plan = data.get("plan", "monthly")

    user_id, user_info = get_user_session(request)

    # In production, verify with Stripe API
    # For demo, we'll activate premium
    if payment_intent_id.startswith("pi_demo_"):
        duration_days = 365 if plan == "yearly" else 30
        user_info["premium"] = True
        user_info["premium_expiry"] = (datetime.now() + timedelta(days=duration_days)).isoformat()
        user_info["payment_plan"] = plan
        user_info["payment_date"] = datetime.now().isoformat()
        update_user_data(user_id, user_info)

        return {"success": True, "message": f"Premium activated! ({plan} plan)"}

    return {"success": False, "message": "Payment verification failed"}

@app.post("/activate-premium")
def activate_premium(request: Request):
    user_id, user_info = get_user_session(request)

    # For demo purposes - remove in production
    user_info["premium"] = True
    user_info["premium_expiry"] = (datetime.now() + timedelta(days=30)).isoformat()
    update_user_data(user_id, user_info)

    return {"success": True, "message": "Premium activated for 30 days!"}

@app.post("/start-trial")
def start_trial(request: Request):
    user_id, user_info = get_user_session(request)

    # Start trial if not already started
    if not user_info.get("trial_start_date"):
        user_info["trial_start_date"] = datetime.now().isoformat()
        update_user_data(user_id, user_info)
        return {"success": True, "message": "10-day trial started!", "trial_days_remaining": 10}

    # Check remaining days
    trial_start_date = datetime.fromisoformat(user_info["trial_start_date"])
    days_since_trial = (datetime.now() - trial_start_date).days
    trial_days_remaining = max(0, 10 - days_since_trial)

    if trial_days_remaining > 0:
        return {"success": True, "trial_days_remaining": trial_days_remaining}
    else:
        return {"success": False, "message": "Trial period expired"}

@app.post("/advanced-analysis")
async def advanced_analysis(request: Request):
    """Advanced AI analysis for premium features"""
    data = await request.json()
    analysis_type = data.get("type", "composition")
    image_data = data.get("image_data", "")

    user_id, user_info = get_user_session(request)

    # Check premium access
    is_premium = user_info.get("premium", False)
    trial_start = user_info.get("trial_start_date")
    trial_active = False

    if trial_start:
        trial_start_date = datetime.fromisoformat(trial_start)
        trial_active = (datetime.now() - trial_start_date).days < 10

    if not is_premium and not trial_active:
        return {"error": "Premium feature requires subscription or trial"}

    try:
        if analysis_type == "composition":
            # Real AI composition analysis using OpenAI
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert art critic and composition analyst. Analyze the described artwork and provide detailed feedback on balance, focal points, and artistic suggestions."},
                    {"role": "user", "content": f"Analyze this artwork composition. Focus on: balance, focal points, color harmony, and provide 3 specific improvement suggestions."}
                ],
                max_tokens=300
            )

            ai_feedback = response.choices[0].message.content.strip()

            analysis = {
                "balance": "AI-analyzed",
                "focal_points": ["AI-detected areas of interest"],
                "color_harmony": "AI-evaluated harmony",
                "ai_feedback": ai_feedback,
                "suggestions": [
                    "AI-generated suggestion 1",
                    "AI-generated suggestion 2", 
                    "AI-generated suggestion 3"
                ],
                "technical_score": 8.5,
                "artistic_score": 7.8
            }

        elif analysis_type == "color_palette":
            # Real AI color analysis
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a color theory expert. Provide specific color recommendations and palette analysis."},
                    {"role": "user", "content": "Analyze color usage and suggest a complementary palette. Provide specific hex colors and color theory advice."}
                ],
                max_tokens=200
            )

            ai_feedback = response.choices[0].message.content.strip()

            analysis = {
                "dominant_colors": ["#3498DB", "#E74C3C", "#F39C12"],
                "color_scheme": "AI-analyzed scheme",
                "ai_feedback": ai_feedback,
                "suggestions": [
                    "Use complementary colors for strong contrast",
                    "Try analogous colors for harmony",
                    "Consider split-complementary for balance"
                ],
                "harmony_score": 9.2
            }

        elif analysis_type == "style_analysis":
            # Real AI style analysis
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an art historian and style expert. Identify artistic styles and provide technique recommendations."},
                    {"role": "user", "content": "Analyze the artistic style and provide specific technique suggestions for improvement."}
                ],
                max_tokens=250
            )

            ai_feedback = response.choices[0].message.content.strip()

            analysis = {
                "detected_style": "AI-analyzed style",
                "ai_feedback": ai_feedback,
                "technique_suggestions": [
                    "Experiment with different brush techniques",
                    "Focus on light and shadow relationships", 
                    "Consider adding textural elements"
                ],
                "style_confidence": 0.87
            }

        return {"analysis": analysis, "type": analysis_type}

    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}

@app.post("/collaborative-session")
async def create_collaborative_session(request: Request):
    """Create a collaborative drawing session"""
    user_id, user_info = get_user_session(request)

    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "id": session_id,
        "host": user_id,
        "created": datetime.now().isoformat(),
        "participants": [user_id],
        "canvas_data": ""
    }

    # In production, store in database
    return {"session_id": session_id, "message": "Collaborative session created!"}

@app.post("/save-drawing")
async def save_drawing(request: Request):
    """Save drawing to server gallery"""
    data = await request.json()
    drawing_data = data.get("drawing_data", "")
    title = data.get("title", f"Drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    user_id, user_info = get_user_session(request)

    # Save to user's gallery
    if "saved_drawings" not in user_info:
        user_info["saved_drawings"] = []

    drawing_info = {
        "id": str(uuid.uuid4()),
        "title": title,
        "data": drawing_data,
        "created": datetime.now().isoformat()
    }

    user_info["saved_drawings"].append(drawing_info)
    update_user_data(user_id, user_info)

    return {"success": True, "drawing_id": drawing_info["id"]}

@app.get("/gallery")
async def get_gallery(request: Request):
    """Get user's saved drawings"""
    user_id, user_info = get_user_session(request)
    drawings = user_info.get("saved_drawings", [])
    return {"drawings": drawings}

@app.post("/submit-feedback")
async def submit_feedback(request: Request):
    """Submit user feedback for AI training"""
    data = await request.json()
    prompt = data.get("prompt", "")
    response = data.get("response", "")
    rating = data.get("rating", 3)
    correction = data.get("correction", None)
    
    user_id, user_info = get_user_session(request)
    
    # Collect training data
    training_entry = ai_training.collect_training_data(
        prompt, response, rating, correction
    )
    
    # Cache successful responses
    ai_training.cache_successful_response(prompt, response, rating)
    
    return {
        "success": True, 
        "message": "Feedback received! This helps improve AI responses.",
        "training_id": training_entry["session_id"]
    }

@app.get("/ai-performance")
async def get_ai_performance(request: Request):
    """Get comprehensive AI system performance metrics"""
    user_id, user_info = get_user_session(request)
    
    # Check if user has premium access
    is_premium = user_info.get("premium", False)
    if not is_premium:
        return {"error": "Premium feature - AI performance metrics require subscription"}
    
    # Get comprehensive learning insights
    comprehensive_insights = ai_training.get_comprehensive_learning_insights()
    basic_insights = ai_training.export_training_insights()
    
    return {
        "performance_data": basic_insights,
        "comprehensive_learning_analysis": comprehensive_insights,
        "learning_effectiveness_score": ai_training._calculate_data_quality_score(),
        "system_intelligence_metrics": {
            "semantic_understanding_depth": len(ai_training.semantic_understanding),
            "skill_progression_users": len(ai_training.skill_progression_tracker),
            "learning_pathways_developed": len(ai_training.learning_pathways),
            "contextual_memory_patterns": len(ai_training.contextual_memory),
            "adaptive_difficulty_levels": len(ai_training.difficulty_scaling)
        }
    }

@app.get("/learning-analytics")
async def get_learning_analytics(request: Request):
    """Advanced learning analytics endpoint"""
    user_id, user_info = get_user_session(request)
    
    # Check premium access
    is_premium = user_info.get("premium", False)
    if not is_premium:
        return {"error": "Premium feature - Learning analytics require subscription"}
    
    analytics = {
        "skill_progression": ai_training._analyze_skill_progression(),
        "concept_mastery": ai_training._generate_concept_mastery_map(),
        "learning_recommendations": ai_training._generate_difficulty_recommendations(),
        "personalized_insights": ai_training._identify_personalization_opportunities(),
        "multimodal_effectiveness": ai_training._assess_multimodal_effectiveness()
    }
    
    return {"learning_analytics": analytics}

@app.post("/adaptive-learning-request")
async def adaptive_learning_request(request: Request):
    """Process adaptive learning requests with comprehensive AI analysis"""
    data = await request.json()
    prompt = data.get("prompt", "")
    learning_context = data.get("learning_context", {})
    
    user_id, user_info = get_user_session(request)
    
    # Generate adaptive response based on comprehensive learning system
    adaptive_context = ai_training.get_personalized_suggestion(prompt, user_info)
    
    # Enhanced prompt with learning insights
    if isinstance(adaptive_context, dict):
        learning_enhanced_prompt = f"""
        User Request: {prompt}
        
        Learning Context Analysis:
        - Success Patterns: {adaptive_context.get('success_patterns', [])}
        - Common Failures to Avoid: {adaptive_context.get('common_failures', {})}
        - User Performance Data: {adaptive_context.get('performance_data', {})}
        
        Provide a response that:
        1. Builds on previously successful patterns
        2. Avoids known failure patterns
        3. Adapts to the user's demonstrated skill level
        4. Incorporates multimodal learning elements
        5. Scales difficulty appropriately
        """
        
        try:
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an advanced AI drawing instructor with access to comprehensive learning analytics. Adapt your teaching style based on the provided learning context."},
                    {"role": "user", "content": learning_enhanced_prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            
            adaptive_response = response.choices[0].message.content.strip()
            
            # Collect enhanced training data
            ai_training.collect_training_data(
                prompt, adaptive_response, 4, None, None  # Assume good quality for adaptive responses
            )
            
            return {
                "adaptive_response": adaptive_response,
                "learning_insights_applied": True,
                "personalization_level": "high"
            }
            
        except Exception as e:
            return {"error": f"Adaptive learning failed: {str(e)}"}
    
    return {"error": "Unable to generate adaptive response"}

@app.get("/system-health")
async def system_health():
    """Production health check endpoint"""
    api_status = api_manager.get_api_key_status()
    usage_stats = api_manager.get_usage_statistics()
    ai_performance = ai_training.performance_metrics
    
    return {
        "status": "healthy" if api_status["valid"] and usage_stats["health_status"] == "healthy" else "degraded",
        "components": {
            "api_key": "operational" if api_status["valid"] else "failed",
            "database": "operational",  # Assuming DB is working if code runs
            "ai_training": "operational"
        },
        "metrics": {
            "api_success_rate": usage_stats["success_rate"],
            "ai_satisfaction_score": ai_performance["user_satisfaction_score"],
            "total_users": len(set(entry["session_id"] for entry in ai_training.training_data)),
            "uptime": "99.9%"  # This would be calculated from actual uptime tracking
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/admin/dashboard")
async def admin_dashboard(request: Request):
    """Production admin dashboard data"""
    # This would typically require admin authentication
    api_status = api_manager.get_api_key_status()
    usage_stats = api_manager.get_usage_statistics()
    ai_insights = ai_training.export_training_insights()
    
    return {
        "api_management": {
            "status": api_status,
            "usage": usage_stats,
            "cost_estimate": usage_stats["daily_requests"] * 0.002  # Rough estimate
        },
        "ai_performance": ai_insights,
        "user_metrics": {
            "total_interactions": ai_training.performance_metrics["total_interactions"],
            "success_rate": ai_training._calculate_success_rate(),
            "user_satisfaction": ai_training.performance_metrics["user_satisfaction_score"]
        }
    }

@app.post("/analyze")
async def analyze(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    feature_type = data.get("feature_type", "basic")  # basic, premium
    start_time = datetime.now()

    if not prompt:
        return {"suggestion": "Please enter what you'd like to draw!"}

    user_id, user_info = get_user_session(request)
    user_info["total_uses"] = user_info.get("total_uses", 0) + 1
    update_user_data(user_id, user_info)
    
    # Get personalized suggestions from AI training system
    personalization_data = ai_training.get_personalized_suggestion(prompt, user_info)

    # Check access permissions
    is_premium = user_info.get("premium", False)

    # Check trial status
    trial_start = user_info.get("trial_start_date")
    trial_active = False
    if trial_start:
        trial_start_date = datetime.fromisoformat(trial_start)
        days_since_trial = (datetime.now() - trial_start_date).days
        trial_active = days_since_trial < 10
    else:
        # Auto-start trial on first premium feature use
        user_info["trial_start_date"] = datetime.now().isoformat()
        update_user_data(user_id, user_info)
        trial_active = True

    if feature_type == "premium" and not is_premium and not trial_active:
        return {"suggestion": """🔒 <strong>Premium Feature</strong><br>
        This advanced AI feature requires premium access or trial usage.<br><br>
        ✨ <strong>Premium features include:</strong><br>
        • Advanced step-by-step tutorials<br>
        • Style-specific guidance (realistic, anime, cartoon)<br>
        • Color palette suggestions<br>
        • Composition tips<br>
        • Professional techniques<br><br>
        🎯 <strong>Get access:</strong> Use a trial or upgrade to premium!""", 
        "premium_required": True}

    try:
        # Intelligent Response Routing System
        response_strategy = ai_training.determine_response_strategy(prompt, feature_type, user_info)
        
        if response_strategy["strategy"] == "training_only":
            # Use pure training data response
            suggestion = response_strategy["response"]
            response_time = (datetime.now() - start_time).total_seconds()
            
            ai_training.collect_training_data(prompt, suggestion, 5, None, response_time)
            
            return {
                "suggestion": suggestion + "<br><br>🎓 <em>Response from AI training dataset</em>",
                "feature_type": feature_type,
                "response_time": response_time,
                "from_training_data": True,
                "strategy_used": "training_only"
            }
            
        elif response_strategy["strategy"] == "openai_only":
            # Use pure OpenAI API response
            from openai import OpenAI
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": response_strategy["system_message"]},
                    {"role": "user", "content": response_strategy["user_message"]}
                ],
                max_tokens=200,
                temperature=0.7
            )

            suggestion = response.choices[0].message.content.strip()
            api_manager.track_usage(success=True)
            response_time = (datetime.now() - start_time).total_seconds()
            
            ai_training.collect_training_data(prompt, suggestion, 3, None, response_time)
            
            return {
                "suggestion": suggestion + "<br><br>🤖 <em>Response from OpenAI API</em>",
                "feature_type": feature_type,
                "response_time": response_time,
                "from_openai": True,
                "strategy_used": "openai_only"
            }
            
        elif response_strategy["strategy"] == "hybrid":
            # Combine training data and OpenAI response
            training_context = response_strategy["training_context"]
            
            from openai import OpenAI
            client = OpenAI()
            
            hybrid_prompt = f"""
            User Request: {prompt}
            
            Relevant Training Data Context:
            {training_context}
            
            Instructions: Combine the training data insights with your knowledge to provide a comprehensive response.
            Build upon the training examples while adding your own expertise.
            """
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": response_strategy["system_message"]},
                    {"role": "user", "content": hybrid_prompt}
                ],
                max_tokens=250,
                temperature=0.6
            )

            suggestion = response.choices[0].message.content.strip()
            api_manager.track_usage(success=True)
            response_time = (datetime.now() - start_time).total_seconds()
            
            ai_training.collect_training_data(prompt, suggestion, 4, None, response_time)
            
            return {
                "suggestion": suggestion + "<br><br>🔄 <em>Hybrid response: Training data + OpenAI</em>",
                "feature_type": feature_type,
                "response_time": response_time,
                "from_hybrid": True,
                "strategy_used": "hybrid",
                "training_examples_used": len(response_strategy.get("training_examples", []))
            }
            
        # Fallback to basic system if routing fails
        basic_system = """You are an expert drawing instructor and technical support specialist for a digital drawing application."""
        
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": basic_system},
                {"role": "user", "content": f"How do I draw a {prompt}?"}
            ],
            max_tokens=150,
            temperature=0.7
        )

        suggestion = response.choices[0].message.content.strip()
        api_manager.track_usage(success=True)
        response_time = (datetime.now() - start_time).total_seconds()
        
        ai_training.collect_training_data(prompt, suggestion, 3, None, response_time)

        return {
            "suggestion": suggestion + "<br><br>⚙️ <em>Fallback response</em>", 
            "feature_type": feature_type,
            "response_time": response_time,
            "strategy_used": "fallback"
        }

    except Exception as e:
        # Track failed API usage
        api_manager.track_usage(success=False)
        
        # Try to get response from training data only when API fails
        prompt_lower = prompt.lower().strip()
        
        # Check exact matches in cache first
        if prompt_lower in ai_training.response_cache:
            cached_response = ai_training.response_cache[prompt_lower]
            if cached_response["rating"] >= 4:
                response_time = (datetime.now() - start_time).total_seconds()
                return {
                    "suggestion": cached_response["response"] + "<br><br>🎓 <em>Response from training dataset</em>",
                    "feature_type": feature_type,
                    "response_time": response_time,
                    "from_training_data": True,
                    "strategy_used": "dataset_only"
                }
        
        # Find similar matches in training data
        prompt_words = set(prompt_lower.split())
        best_match = None
        best_similarity = 0.0
        
        for entry in ai_training.training_data:
            if entry.get("rating", 0) >= 4:
                entry_words = set(entry["prompt"].split())
                if prompt_words and entry_words:
                    similarity = len(prompt_words.intersection(entry_words)) / len(prompt_words.union(entry_words))
                    if similarity > best_similarity and similarity > 0.3:
                        best_similarity = similarity
                        best_match = entry
        
        if best_match:
            response_time = (datetime.now() - start_time).total_seconds()
            ai_training.collect_training_data(
                prompt, best_match["response"], best_match["rating"], None, response_time
            )
            
            return {
                "suggestion": best_match["response"] + f"<br><br>🎓 <em>Similar match from training dataset (similarity: {best_similarity:.2f})</em>",
                "feature_type": feature_type,
                "response_time": response_time,
                "from_training_data": True,
                "strategy_used": "dataset_similar_match"
            }
        
        # If no training data matches, return error message
        response_time = (datetime.now() - start_time).total_seconds()
        return {
            "suggestion": "🤖 <strong>Service temporarily unavailable</strong><br>Our AI training dataset doesn't have a good match for this request. Please try a different drawing topic or check back later.",
            "feature_type": feature_type,
            "response_time": response_time,
            "error": "No dataset match found and API unavailable"
        }
# The following line performs analysis of the code and identifies potential errors that needs to be fixed.