# Báo cáo giữa kỳ  Chủ đề 7: MIMO-OFDM & LMMSE
- Hình: figs/awgn_64qam.png, figs/ofdm_64qam.png, figs/mimo_zf_64qam.png, figs/mimo_lmmse_64qam.png
- Tổng hợp: figs/compare_ber.png ; Bảng: results/ber_summary.csv
- Cấu hình: exp/*.yml (ghi seed, target_bit_errors, max_bits_per_point nếu dùng)
- Nhận xét ngắn:
  * OFDM lệch ~1.4 dB so với AWGN (hiệu suất tải & CP)
  * MIMO: LMMSE tốt hơn ZF khoảng X dB @ BER=1e-3 (xem ber_summary.csv)
