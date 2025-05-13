  library(wavelets)
  library(ggplot2)
  library(reshape2)

  freq <- 1200  # Number of sawtooth cycles over the time range
  t <- seq(0, 2*pi, length.out = 512)
  #Sawtooth
  signal <- 2 * ((freq * t / (2*pi)) - floor(0.5 + freq * t / (2*pi)))

  #Sine
  #signal <- sin(freq*t)

  # Perform 4-level DWT
  dwt_result <- dwt(signal,
                    filter = "la8",
                    n.levels = 4,
                    boundary = "periodic")

  # Reconstruct full signal
  reconstructed <- list()
  reconstructed[["Reconstruction"]] <- idwt(dwt_result)

  # Extract detail-only reconstructions
  for (i in 1:4) {
    temp <- dwt_result
    # Zero out all details except level i
    for (j in 1:4) {
      if (j != i) temp@W[[j]] <- temp@W[[j]] * 0
    }
    # Zero out approximations
    temp@W[[j]] <- temp@W[[j]] * 0
    reconstructed[[paste0("Detail", i)]] <- idwt(temp)
  }

  # Combine into data frame manually to avoid melt issues
  df <- data.frame(
    Time = t[1:length(reconstructed[[1]])],  # handles shorter vectors if needed
    Original = signal[1:length(reconstructed[[1]])]
  )

  for (name in names(reconstructed)) {
    df[[name]] <- reconstructed[[name]]
  }

  # Reshape for plotting
  df_long <- reshape2::melt(df, id.vars = "Time")

  # Plot
  ggplot(df_long, aes(x = Time, y = value)) +
    geom_line() +
    facet_wrap(~ variable, scales = "free_y", ncol = 1) +
    theme_minimal() +
    labs(title = "Wavelet Transform Decomposition of Sine Wave",
         x = "Time", y = "Amplitude") +
    theme(legend.position = "none")
