# TogetherAI Rate Limiting Configuration

## Overview

This service implements tier-based rate limiting for TogetherAI API calls to respect their usage limits and prevent 429 (Too Many Requests) errors.

## TogetherAI Build Tiers

| Tier | Total Spend | LLM RPM | Embeddings RPM | Re-rank RPM | Vision Concurrency* |
|------|-------------|---------|----------------|-------------|---------------------|
| 1    | $5.00       | 600     | 3,000          | 500,000     | 12                  |
| 2    | $50.00      | 1,800   | 5,000          | 1,500,000   | 20                  |
| 3    | $100.00     | 3,000   | 5,000          | 2,000,000   | 20                  |
| 4    | $250.00     | 4,500   | 10,000         | 3,000,000   | 20                  |
| 5    | $1,000.00   | 6,000   | 10,000         | 10,000,000  | 20                  |

*Vision Concurrency = Safe number of concurrent vision model requests

## Configuration

### Setting the Tier

The tier is configured via the `TOGETHER_AI_TIER` environment variable:

```bash
# In systemd service file
Environment=TOGETHER_AI_TIER=1

# Or export in shell
export TOGETHER_AI_TIER=1
```

**Default:** Tier 1 (if not specified)

### Current Configuration

The service is currently configured for **Tier 1** ($5 spent).

To change the tier:

1. Edit the systemd service file:
   ```bash
   sudo nano /etc/systemd/system/mivaa-pdf-extractor.service
   ```

2. Update the `TOGETHER_AI_TIER` value:
   ```
   Environment=TOGETHER_AI_TIER=2  # Change to desired tier
   ```

3. Reload and restart:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart mivaa-pdf-extractor.service
   ```

## How It Works

### Vision Concurrency Calculation

The system automatically calculates safe concurrency limits based on the tier:

```python
# Formula
safe_rpm = tier.llm_rpm * 0.6  # Use 60% of limit (40% headroom)
avg_response_time = 2.0  # seconds
concurrent_limit = (safe_rpm / 60) * avg_response_time
concurrent_limit = max(2, min(20, concurrent_limit))  # Clamp 2-20
```

### Examples by Tier

- **Tier 1** (600 RPM): `(600 * 0.6 / 60) * 2 = 12` concurrent requests
- **Tier 2** (1800 RPM): `(1800 * 0.6 / 60) * 2 = 36 → 20` (capped at 20)
- **Tier 3** (3000 RPM): `(3000 * 0.6 / 60) * 2 = 60 → 20` (capped at 20)

### Why 60% of RPM?

- **40% headroom** for retry attempts (3 retries per request)
- **Burst tolerance** for temporary spikes
- **Other API usage** (embeddings, text models)
- **Safety margin** to prevent hitting rate limits

## Monitoring

### Check Current Configuration

View the rate limiting configuration in logs when processing starts:

```bash
journalctl -u mivaa-pdf-extractor.service | grep "Rate Limiting Configuration"
```

Expected output:
```
🎯 Rate Limiting Configuration:
   TogetherAI Tier: 1 ($5.0 spent)
   LLM Rate Limit: 600 RPM (10.0 RPS)
   Vision Concurrency: 12 concurrent requests
   Claude Concurrency: 2 concurrent requests
```

### Monitor API Errors

Check for rate limit errors:

```bash
journalctl -u mivaa-pdf-extractor.service | grep -E "429|rate.limit|service_unavailable"
```

## Error Handling

The service includes robust error handling for API failures:

1. **Error Detection**: Checks for `error` key in API responses
2. **Retry Logic**: 3 attempts with exponential backoff (1s, 2s, 4s)
3. **Graceful Degradation**: Falls back to secondary models on failure
4. **Detailed Logging**: Logs error type, message, and HTTP status

## Files

- `app/config/rate_limits.py` - Rate limit configuration and tier definitions
- `app/config.py` - Application settings including `TOGETHER_AI_TIER`
- `app/services/images/image_processing_service.py` - Implementation

## Upgrading Tiers

As your TogetherAI spend increases, update the tier:

1. Check your current spend in TogetherAI dashboard
2. Determine your new tier from the table above
3. Update `TOGETHER_AI_TIER` environment variable
4. Restart the service

Higher tiers allow more concurrent requests and faster processing.

## Best Practices

1. **Start with Tier 1** - Default conservative settings
2. **Monitor usage** - Watch for 429 errors or slow processing
3. **Upgrade as needed** - Increase tier when you hit limits
4. **Leave headroom** - Don't use 100% of your rate limit
5. **Use retries** - Built-in retry logic handles transient failures

## Troubleshooting

### Too Many 429 Errors

- Current tier may be too low for your usage
- Consider upgrading to next tier
- Check if other services are using the same API key

### Slow Processing

- Increase tier for higher concurrency
- Current tier may be bottlenecking throughput

### Service Unavailable Errors

- TogetherAI API may be experiencing issues
- Retry logic will automatically handle transient failures
- Check TogetherAI status page

## Support

For questions or issues:
- Check logs: `journalctl -u mivaa-pdf-extractor.service -f`
- Review error messages for specific guidance
- Consult TogetherAI documentation: https://docs.together.ai/docs/rate-limits

