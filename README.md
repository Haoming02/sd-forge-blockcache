# SD Forge Block Cache
This is an Extension for [Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic), which implements **First Block Cache** and **Tea Cache**, forked to focus on speeding up `SDXL` checkpoints.

> For running `Flux` checkpoints on the modern [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), use the [original repo](https://github.com/DenOfEquity/sd-forge-blockcache) instead

<details>
<summary>Benchmark</summary>

- Generate a `896x1152` image using `SDXL` checkpoint in `24` Steps on a `RTX 3060`

<table>
    <tbody align="center">
        <tr>
            <td>Extension</td>
            <td>Disabled</td>
            <td>First Block Cache</td>
            <td>Tea Cache</td>
        </tr>
        <tr>
            <td>Cache Start</td>
            <td>n.a.</td>
            <td>0.4</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>Cache Threshold</td>
            <td>n.a.</td>
            <td>0.4</td>
            <td>0.4</td>
        </tr>
        <tr>
            <td>Result</td>
            <td><img src="img\off.jpg" width=256></td>
            <td><img src="img\fb.jpg" width=256></td>
            <td><img src="img\tea.jpg" width=256></td>
        </tr>
        <tr>
            <td>Time</td>
            <td>15s</td>
            <td>12s</td>
            <td>12s</td>
        </tr>
    </tbody>
</table>

</details>

> [!IMPORTANT]
> - This Extension tends to generate noise if the values are set too strong
> - `Sampling method` also has an impact on the effect of this Extension
> - The speed up is more noticeable the more `Steps` is used

<hr>

## Reference

- https://github.com/ali-vilab/TeaCache
- https://github.com/chengzeyi/Comfy-WaveSpeed
- https://github.com/DenOfEquity/sd-forge-blockcache
