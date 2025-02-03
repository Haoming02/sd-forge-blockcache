# SD Forge Block Cache
This is an Extension for [Forge Classic](https://github.com/Haoming02/sd-webui-forge-classic), which implements **First Block Cache** and **Tea Cache**, forked to focus on speeding up `SDXL` checkpoints.

> To run `Flux` checkpoints on the new [Forge Webui](https://github.com/lllyasviel/stable-diffusion-webui-forge), use the [original repo](https://github.com/DenOfEquity/sd-forge-blockcache) instead

<details open>
<summary>Benchmark</summary>

- Generate a `896x1152` image using `SDXL` checkpoint in `64` Steps with `Euler a` sampler on a `RTX 3060`

<table>
    <tbody align="center">
        <tr>
            <td>\</td>
            <td><b>Disabled</b></td>
            <td><b>First Block Cache</b></td>
            <td><b>Tea Cache</b></td>
            <td><b>Tea Cache</b></td>
        </tr>
        <tr>
            <td><b>Cache Start</b></td>
            <td rowspan="3">n.a.</td>
            <td>0.6</td>
            <td>0.4</td>
            <td>0.2</td>
        </tr>
        <tr>
            <td><b>Cache Threshold</b></td>
            <td>0.4</td>
            <td>0.6</td>
            <td>0.8</td>
        </tr>
        <tr>
            <td><b>Cache Limit</b></td>
            <td>0</td>
            <td>0</td>
            <td>1</td>
        </tr>
        <tr>
            <td><b>Result</b></td>
            <td><img src="img\off.jpg" width=256></td>
            <td><img src="img\fb.jpg" width=256></td>
            <td><img src="img\tea.jpg" width=256></td>
            <td><img src="img\tea1.jpg" width=256></td>
        </tr>
        <tr>
            <td><b>Time</b></td>
            <td>45s</td>
            <td>35s</td>
            <td>25s</td>
            <td>25s</td>
        </tr>
    </tbody>
</table>

</details>

> [!IMPORTANT]
> The Extension may result in some noisy patterns to appear on the images, with certain `Sampling method` causing worse effects; though using `Cache Limit` can help mitigate this, and doing another pass *(**eg.** `Hires. fix` or `img2img`)* usually cleans them up as well

<hr>

## Special Thanks

- https://github.com/DenOfEquity/sd-forge-blockcache
- https://github.com/chengzeyi/Comfy-WaveSpeed
- https://github.com/ali-vilab/TeaCache
