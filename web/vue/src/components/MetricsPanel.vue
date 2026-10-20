<template>
  <aside class="metrics-panel">
    <div class="panel-header">
      <span class="panel-title">性能指标</span>
      <el-button text :icon="'Close'" size="small" @click="$emit('close')" />
    </div>

    <div v-if="metrics" class="metrics-content">
      <!-- 总览 -->
      <div class="metric-section">
        <h4 class="section-title">总览</h4>
        <div class="metric-grid">
          <div class="metric-card">
            <span class="metric-value">{{ metrics.total_tokens ?? '-' }}</span>
            <span class="metric-label">总 Token</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{{ formatDuration(metrics.duration_seconds) }}</span>
            <span class="metric-label">耗时</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{{ metrics.total_chars ?? '-' }}</span>
            <span class="metric-label">总字符</span>
          </div>
          <div class="metric-card">
            <span class="metric-value">{{ metrics.total_messages ?? '-' }}</span>
            <span class="metric-label">消息数</span>
          </div>
        </div>
        <!-- Token 分解 -->
        <div v-if="metrics.prompt_tokens || metrics.completion_tokens" class="token-breakdown">
          <div class="breakdown-row">
            <span class="breakdown-label">Prompt</span>
            <span class="breakdown-val">{{ metrics.prompt_tokens ?? 0 }}</span>
          </div>
          <div class="breakdown-row">
            <span class="breakdown-label">Completion</span>
            <span class="breakdown-val">{{ metrics.completion_tokens ?? 0 }}</span>
          </div>
        </div>
      </div>

      <!-- Agent 分步 -->
      <div v-if="metrics.agent_metrics && metrics.agent_metrics.length" class="metric-section">
        <h4 class="section-title">Agent 分步</h4>
        <div class="agent-metrics-list">
          <div
            v-for="(am, idx) in metrics.agent_metrics"
            :key="idx"
            class="agent-metric-row"
          >
            <el-tag size="small" :type="agentType(am.name)" effect="dark">{{ am.name }}</el-tag>
            <span class="agent-metric-val">{{ am.tokens ?? '-' }} tok</span>
            <span class="agent-metric-val">{{ formatDuration(am.duration) }}</span>
            <span class="agent-metric-val" v-if="am.memory_hits">{{ am.memory_hits }} hits</span>
            <span class="agent-metric-val" v-if="am.precise_chars">{{ am.precise_chars }} chars</span>
          </div>
        </div>
      </div>

      <div v-if="metrics.summary?.memory_query_count != null" class="metric-section">
        <h4 class="section-title">跨任务记忆</h4>
        <div class="token-breakdown">
          <div class="breakdown-row"><span>检索命中率</span><span>{{ ((metrics.memory_hit_rate ?? 0) * 100).toFixed(1) }}%</span></div>
          <div class="breakdown-row"><span>历史记忆返回数</span><span>{{ metrics.summary.cross_task_memory_hits ?? 0 }}</span></div>
          <div class="breakdown-row"><span>历史记忆引用数</span><span>{{ metrics.summary.cross_task_memory_citations ?? 0 }}</span></div>
          <div class="breakdown-row"><span>向量编码缓存命中</span><span>{{ metrics.summary.embedding_cache_hits ?? 0 }}</span></div>
          <div class="breakdown-row"><span>记忆检索向量载荷</span><span>{{ metrics.summary.memory_state_bytes ?? 0 }} B</span></div>
          <div class="breakdown-row"><span>记忆整理耗时</span><span>{{ formatDuration(metrics.summary.memory_consolidation_seconds) }}</span></div>
        </div>
        <p class="metric-label">引用数按模型返回的有效记忆 ID 统计；正确性需另行评测。向量载荷不含传输协议开销。</p>
        <p v-if="metrics.summary.memory_consolidation_usage_unknown" class="metric-label">记忆整理失败，相关调用的 Token 用量未知。</p>
      </div>

      <!-- 三层协议 -->
      <div v-if="metrics.protocol_metrics" class="metric-section">
        <h4 class="section-title">三层协议</h4>
        <div class="protocol-list">
          <div class="protocol-row">
            <span class="protocol-label">Control</span>
            <span class="protocol-val">{{ metrics.protocol_metrics.control_tokens ?? 0 }} tok / {{ metrics.protocol_metrics.control_decisions ?? 0 }} decisions</span>
          </div>
          <div class="protocol-row">
            <span class="protocol-label">Semantic</span>
            <span class="protocol-val">{{ metrics.protocol_metrics.semantic_tokens ?? 0 }} tok / {{ metrics.protocol_metrics.semantic_transfers ?? 0 }} transfers / {{ metrics.protocol_metrics.semantic_bytes ?? 0 }} B</span>
          </div>
          <div class="protocol-row">
            <span class="protocol-label">Precise</span>
            <span class="protocol-val">{{ metrics.protocol_metrics.precise_tokens ?? 0 }} tok / {{ metrics.protocol_metrics.precise_chars ?? 0 }} chars</span>
          </div>
        </div>
      </div>
    </div>

    <div v-else class="empty-metrics">
      <el-icon :size="32"><DataLine /></el-icon>
      <span>发送消息后查看指标</span>
    </div>
  </aside>
</template>

<script setup>
import { DataLine } from '@element-plus/icons-vue'

defineProps({
  metrics: { type: Object, default: null },
})

defineEmits(['close'])

function formatDuration(seconds) {
  if (!seconds) return '-'
  return seconds.toFixed(2) + 's'
}

function agentType(name) {
  const map = { 任务分配: 'primary', 资料搜索: 'success', 任务执行: 'warning', 总结验证: 'danger' }
  return map[name] || 'info'
}
</script>

<style scoped>
.metrics-panel {
  width: 280px;
  height: 100vh;
  display: flex;
  flex-direction: column;
  background: var(--sf-bg-overlay);
  border-left: 1px solid var(--sf-border-light);
  flex-shrink: 0;
  overflow-y: auto;
}

.panel-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid var(--sf-border-light);
}

.panel-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--sf-text-primary);
}

.metrics-content {
  padding: 12px 16px;
}

.metric-section {
  margin-bottom: 16px;
}

.section-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--sf-text-muted);
  text-transform: uppercase;
  margin-bottom: 8px;
  letter-spacing: 0.5px;
}

.metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
}

.metric-card {
  background: var(--sf-bg-elevated);
  border-radius: 8px;
  padding: 10px;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.metric-value {
  font-size: 18px;
  font-weight: 700;
  color: var(--sf-primary);
}

.metric-label {
  font-size: 11px;
  color: var(--sf-text-muted);
  margin-top: 2px;
}

.agent-metrics-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.agent-metric-row {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  background: var(--sf-bg-elevated);
  border-radius: 6px;
  font-size: 12px;
}

.agent-metric-val {
  color: var(--sf-text-secondary);
  margin-left: auto;
  font-size: 11px;
}

.token-breakdown {
  margin-top: 8px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.breakdown-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 10px;
  background: var(--sf-bg-elevated);
  border-radius: 4px;
  font-size: 12px;
}

.breakdown-label {
  color: var(--sf-text-muted);
}

.breakdown-val {
  color: var(--sf-text-secondary);
  font-weight: 600;
}

.protocol-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.protocol-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px;
  background: var(--sf-bg-elevated);
  border-radius: 6px;
}

.protocol-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--sf-primary);
  text-transform: uppercase;
}

.protocol-val {
  font-size: 12px;
  color: var(--sf-text-secondary);
}

.empty-metrics {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  color: var(--sf-text-muted);
  font-size: 13px;
}
</style>
