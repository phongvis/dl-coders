# Chapter 4. Under the Hood: Training a Digit Classifier



```python
from fastai.vision.all import *

matplotlib.rc('image', cmap='Greys')
```

```python
from torch.utils.data import TensorDataset
```

## 1. Baseline binary digit classifier 3 vs. 7
*Practicing PyTorch and build a simple classifier based on similarity with two typical digits.*

```python
path = untar_data(URLs.MNIST_SAMPLE)
Path.BASE_PATH = path
path.ls()
```




    (#3) [Path('valid'),Path('labels.csv'),Path('train')]



```python
def load_digit(set, d):
    files = (path/set/d).ls()
    
    img = Image.open(files[0])
    display(img)
    df = pd.DataFrame(tensor(img)[3:23, 3:23])
    display(df)

    tensors = [tensor(Image.open(f)) for f in files]
    return torch.stack(tensors).float() / 255
```

```python
tensors3 = load_digit('train', '3')
tensors7 = load_digit('train', '7')
tensors3.shape, tensors7.shape
```


![png](/images/04_main_files/output_5_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>104</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>121</td>
      <td>121</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>156</td>
      <td>252</td>
      <td>252</td>
      <td>253</td>
      <td>252</td>
      <td>247</td>
      <td>240</td>
      <td>240</td>
      <td>240</td>
      <td>148</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>26</td>
      <td>26</td>
      <td>159</td>
      <td>158</td>
      <td>158</td>
      <td>158</td>
      <td>233</td>
      <td>252</td>
      <td>252</td>
      <td>212</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>31</td>
      <td>99</td>
      <td>210</td>
      <td>239</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>53</td>
      <td>204</td>
      <td>252</td>
      <td>204</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>49</td>
      <td>166</td>
      <td>238</td>
      <td>241</td>
      <td>198</td>
      <td>143</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>97</td>
      <td>244</td>
      <td>252</td>
      <td>252</td>
      <td>205</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>27</td>
      <td>204</td>
      <td>252</td>
      <td>252</td>
      <td>229</td>
      <td>190</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>26</td>
      <td>252</td>
      <td>253</td>
      <td>246</td>
      <td>238</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>196</td>
      <td>252</td>
      <td>253</td>
      <td>145</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>78</td>
      <td>197</td>
      <td>255</td>
      <td>253</td>
      <td>133</td>
      <td>83</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>77</td>
      <td>146</td>
      <td>247</td>
      <td>252</td>
      <td>248</td>
      <td>134</td>
      <td>85</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>151</td>
      <td>158</td>
      <td>237</td>
      <td>252</td>
      <td>218</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>145</td>
      <td>193</td>
      <td>252</td>
      <td>133</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>137</td>
      <td>252</td>
      <td>231</td>
      <td>44</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>88</td>
      <td>204</td>
      <td>252</td>
      <td>222</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>17</td>
      <td>160</td>
      <td>240</td>
      <td>252</td>
      <td>194</td>
      <td>67</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>57</td>
      <td>191</td>
      <td>111</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>27</td>
      <td>98</td>
      <td>161</td>
      <td>199</td>
      <td>252</td>
      <td>243</td>
      <td>120</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>57</td>
      <td>240</td>
      <td>252</td>
      <td>252</td>
      <td>252</td>
      <td>252</td>
      <td>252</td>
      <td>252</td>
      <td>252</td>
      <td>253</td>
      <td>252</td>
      <td>190</td>
      <td>72</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



![png](/images/04_main_files/output_5_2.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>64</td>
      <td>145</td>
      <td>161</td>
      <td>221</td>
      <td>254</td>
      <td>138</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>214</td>
      <td>230</td>
      <td>231</td>
      <td>180</td>
      <td>138</td>
      <td>138</td>
      <td>139</td>
      <td>138</td>
      <td>214</td>
      <td>230</td>
      <td>231</td>
      <td>251</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>137</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>146</td>
      <td>230</td>
      <td>245</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>230</td>
      <td>230</td>
      <td>154</td>
      <td>104</td>
      <td>254</td>
      <td>253</td>
      <td>137</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>69</td>
      <td>111</td>
      <td>128</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>254</td>
      <td>253</td>
      <td>71</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>93</td>
      <td>255</td>
      <td>254</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>176</td>
      <td>254</td>
      <td>253</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184</td>
      <td>254</td>
      <td>215</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>184</td>
      <td>254</td>
      <td>139</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>254</td>
      <td>254</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>253</td>
      <td>254</td>
      <td>115</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>253</td>
      <td>254</td>
      <td>107</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>149</td>
      <td>253</td>
      <td>254</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>208</td>
      <td>254</td>
      <td>185</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>207</td>
      <td>253</td>
      <td>184</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>207</td>
      <td>253</td>
      <td>109</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>240</td>
      <td>253</td>
      <td>93</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





    (torch.Size([6131, 28, 28]), torch.Size([6265, 28, 28]))



```python
mean3 = tensors3.mean(0)
mean7 = tensors7.mean(0)
show_image(mean3)
show_image(mean7)
```




    <AxesSubplot:>




![png](/images/04_main_files/output_6_1.png)



![png](/images/04_main_files/output_6_2.png)


```python
def dist(a, b):
    # a, b can have different ranks, broadcasting
    return (a - b).abs().mean((-1, -2))

def is3(x):
    return dist(x, mean3) < dist(x, mean7)
```

```python
is3(tensors3[0]), is3(tensors7[0])
```




    (tensor(True), tensor(False))



### Compute accuracy for the validation set

```python
valid_tensors3 = load_digit('valid', '3')
valid_tensors7 = load_digit('valid', '7')
valid_tensors3.shape, valid_tensors7.shape
```


![png](/images/04_main_files/output_10_0.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>111</td>
      <td>155</td>
      <td>174</td>
      <td>254</td>
      <td>254</td>
      <td>254</td>
      <td>188</td>
      <td>101</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>163</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>249</td>
      <td>69</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>85</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>20</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>79</td>
      <td>252</td>
      <td>253</td>
      <td>242</td>
      <td>153</td>
      <td>133</td>
      <td>133</td>
      <td>167</td>
      <td>253</td>
      <td>253</td>
      <td>106</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>79</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>253</td>
      <td>253</td>
      <td>148</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>253</td>
      <td>253</td>
      <td>218</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>39</td>
      <td>170</td>
      <td>253</td>
      <td>253</td>
      <td>218</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>24</td>
      <td>121</td>
      <td>202</td>
      <td>246</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>125</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>150</td>
      <td>194</td>
      <td>229</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>208</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>77</td>
      <td>184</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>180</td>
      <td>22</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>171</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>251</td>
      <td>221</td>
      <td>198</td>
      <td>250</td>
      <td>253</td>
      <td>253</td>
      <td>109</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>18</td>
      <td>235</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>216</td>
      <td>107</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>218</td>
      <td>253</td>
      <td>171</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>42</td>
      <td>197</td>
      <td>253</td>
      <td>251</td>
      <td>238</td>
      <td>140</td>
      <td>59</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>134</td>
      <td>253</td>
      <td>229</td>
      <td>25</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>55</td>
      <td>183</td>
      <td>77</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>162</td>
      <td>253</td>
      <td>253</td>
      <td>54</td>
    </tr>
    <tr>
      <th>14</th>
      <td>28</td>
      <td>82</td>
      <td>26</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7</td>
      <td>75</td>
      <td>195</td>
      <td>251</td>
      <td>254</td>
      <td>231</td>
      <td>28</td>
    </tr>
    <tr>
      <th>15</th>
      <td>38</td>
      <td>240</td>
      <td>253</td>
      <td>189</td>
      <td>179</td>
      <td>90</td>
      <td>54</td>
      <td>13</td>
      <td>51</td>
      <td>48</td>
      <td>80</td>
      <td>80</td>
      <td>126</td>
      <td>200</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>177</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>143</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>247</td>
      <td>237</td>
      <td>246</td>
      <td>246</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>254</td>
      <td>241</td>
      <td>55</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>3</td>
      <td>187</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>192</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>185</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>241</td>
      <td>159</td>
      <td>109</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>54</td>
      <td>106</td>
      <td>153</td>
      <td>153</td>
      <td>230</td>
      <td>253</td>
      <td>212</td>
      <td>153</td>
      <td>109</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



![png](/images/04_main_files/output_10_2.png)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
      <th>14</th>
      <th>15</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>116</td>
      <td>254</td>
      <td>229</td>
      <td>161</td>
      <td>161</td>
      <td>161</td>
      <td>162</td>
      <td>161</td>
      <td>86</td>
      <td>136</td>
      <td>104</td>
      <td>87</td>
      <td>120</td>
      <td>70</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>0</td>
      <td>116</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>142</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>221</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>254</td>
      <td>253</td>
      <td>253</td>
      <td>253</td>
      <td>216</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>94</td>
      <td>77</td>
      <td>111</td>
      <td>160</td>
      <td>161</td>
      <td>143</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>69</td>
      <td>152</td>
      <td>253</td>
      <td>254</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>141</td>
      <td>254</td>
      <td>210</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>232</td>
      <td>253</td>
      <td>167</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>123</td>
      <td>253</td>
      <td>253</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>172</td>
      <td>253</td>
      <td>219</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>237</td>
      <td>254</td>
      <td>161</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>153</td>
      <td>253</td>
      <td>248</td>
      <td>64</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>170</td>
      <td>253</td>
      <td>154</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>34</td>
      <td>254</td>
      <td>253</td>
      <td>137</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>151</td>
      <td>255</td>
      <td>254</td>
      <td>46</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>197</td>
      <td>254</td>
      <td>244</td>
      <td>38</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>99</td>
      <td>253</td>
      <td>254</td>
      <td>131</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





    (torch.Size([1010, 28, 28]), torch.Size([1028, 28, 28]))



```python
acc3 = is3(valid_tensors3).float().mean()
acc7 = 1 - is3(valid_tensors7).float().mean()
acc3, acc7, (acc3 + acc7) / 2
```




    (tensor(0.9168), tensor(0.9854), tensor(0.9511))



## 2. SGD

### Mapping to a ML problem
- Each position in an image contributes to the classification decision. For instance, pixels at the bottom right corners mean it's less likely that the image is a 7. We can consider each position as a **weight**.
- The intensity of each pixel also contributes.
- We can compute some value based on these two elements that can be used to help the classification. And we want to find the weights so that the computed values can be used to classify as accurate as possible.

### Process
1. Initialize the weights.
1. For each image, use these weights to predict whether it appears to be a 3 or a 7.
1. Based on these predictions, calculate how good the model is (its loss).
1. Calculate the gradient, which measures for each weight, how changing that weight would change the loss
1. Step (that is, change) all the weights based on that calculation.
1. Go back to the step 2, and repeat the process.
1. Iterate until you decide to stop the training process (for instance, because the model is good enough or you don't want to wait any longer).

```python
def create_dl(tensors3, tensors7, bs=256, shuffle=False):
    X = torch.cat([tensors3, tensors7]).view(-1, 28*28)
    y = tensor([1.0] * len(tensors3) + [0.0] * len(tensors7)).unsqueeze(1)
    print(X.shape, y.shape)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, bs=bs, shuffle=shuffle)
    return dl
```

```python
dl = create_dl(tensors3, tensors7, bs=256, shuffle=True)
valid_dl = create_dl(valid_tensors3, valid_tensors7, bs=256, shuffle=False)
```

    torch.Size([12396, 784]) torch.Size([12396, 1])
    torch.Size([2038, 784]) torch.Size([2038, 1])


```python
def linear(xb):
    return xb@weights + bias

def validate_epoch(model):
    all_preds = []
    all_targets = []
    for xb, yb in valid_dl:
        all_preds.append(model(xb).sigmoid())
        all_targets.append(yb)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return ((all_preds > 0.5) == all_targets).float().mean().item()
```

### Implementing the process from scratch

```python
# 1. Initialize the weights
weights = torch.randn((28 * 28, 1), requires_grad=True)
bias = torch.randn(1, requires_grad=True)
lr = 1
epochs = 10

for epoch in range(epochs):
    # Train
    for xb, yb in dl:
        # 2. Forward pass, compute predictions
        preds = xb@weights + bias
        
        # 3. Compute loss
        preds = preds.sigmoid().clamp(1e-6, 1 - 1e-6)
        loss = (-yb*torch.log(preds) - (1-yb)*torch.log(1-preds)).mean()
        
        # 4. Compute gradients
        loss.backward()
        
        # 5. Update weights
        with torch.no_grad():
            for p in [weights, bias]:
                p -= p.grad * lr
                p.grad.zero_()

    # Validation     
    print(f'{validate_epoch(linear):.3f}')
```

    0.951
    0.967
    0.973
    0.974
    0.980
    0.982
    0.982
    0.982
    0.983
    0.983


### Implementing the process with PyTorch classes

Simplify:
1. Use `nn.Linear` which does both weights initialization (Step 1) and linear transformation (Step 2).
1. Use `nn.BCEWithLoss` as a loss function (Step 3)
1. Use an optimizer `torch.optim.SGD` which handles step and zero grad (Step 4 and 5).

```python
# 1. Initialize the weights
linear_model = nn.Linear(28 * 28, 1)
optimizer = torch.optim.SGD(linear_model.parameters(), lr=1)
epochs = 10

for epoch in range(epochs):
    # Train
    for xb, yb in dl:
        # 2. Forward pass, compute predictions
        preds = linear_model(xb)
        
        # 3. Compute loss
        loss = nn.BCEWithLogitsLoss()(preds, yb)
        
        # 4. Compute gradients
        loss.backward()
        
        # 5. Update weights
        optimizer.step()
        optimizer.zero_grad()

    # Validation     
    print(f'{validate_epoch(linear_model):.3f}')
```

    0.976
    0.980
    0.980
    0.980
    0.980
    0.981
    0.982
    0.982
    0.982
    0.982


### Wrapping up in a fastai learner class
fastai provides a class to encapsulate the training process. Besides a standard model architecture, a loss function and an optimizer, it requires two extra pieces for validation:
- a `DataLoaders` instance which simply combines train and validation standard data loaders 
- a list of metrics to compute for the validation set at the end of each epoch.

```python
dls = DataLoaders(dl, valid_dl)
```

```python
def batch_accuracy(xb, yb):
    preds = xb.sigmoid()
    correct = (preds > 0.5) == yb
    return correct.float().mean()
```

```python
learn = Learner(dls, nn.Linear(28*28, 1), loss_func=nn.BCEWithLogitsLoss(), opt_func=SGD, metrics=batch_accuracy)
learn.fit(10, lr=1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.079573</td>
      <td>0.067813</td>
      <td>0.977429</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.060828</td>
      <td>0.060476</td>
      <td>0.978410</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.051559</td>
      <td>0.057214</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.046701</td>
      <td>0.063618</td>
      <td>0.980373</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.044764</td>
      <td>0.052497</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.040757</td>
      <td>0.052090</td>
      <td>0.982336</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.038731</td>
      <td>0.053444</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.038150</td>
      <td>0.052197</td>
      <td>0.980864</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.037722</td>
      <td>0.054257</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.036560</td>
      <td>0.049561</td>
      <td>0.982826</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


## 3. Deep neural networks

With the basics above, it's easy to extend the linear model to a (deep) neural network. The only change required is the forward pass in Step 2.

```python
deep_net = nn.Sequential(
    nn.Linear(28*28, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

learn = Learner(dls, deep_net, loss_func=nn.BCEWithLogitsLoss(), opt_func=SGD, metrics=batch_accuracy)
learn.fit(20, lr=0.1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.689883</td>
      <td>0.684060</td>
      <td>0.915604</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.641845</td>
      <td>0.462405</td>
      <td>0.964181</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.308381</td>
      <td>0.076148</td>
      <td>0.972522</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.147456</td>
      <td>0.071248</td>
      <td>0.974975</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.084200</td>
      <td>0.055969</td>
      <td>0.981354</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.057554</td>
      <td>0.051549</td>
      <td>0.984298</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.044683</td>
      <td>0.048667</td>
      <td>0.984298</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.037882</td>
      <td>0.047593</td>
      <td>0.983808</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.036272</td>
      <td>0.043861</td>
      <td>0.985770</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.033067</td>
      <td>0.043448</td>
      <td>0.985770</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.031724</td>
      <td>0.042149</td>
      <td>0.985770</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.029148</td>
      <td>0.067802</td>
      <td>0.980373</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.027495</td>
      <td>0.040559</td>
      <td>0.985770</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.025184</td>
      <td>0.036583</td>
      <td>0.986752</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.023736</td>
      <td>0.043045</td>
      <td>0.985280</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>15</td>
      <td>0.021933</td>
      <td>0.033754</td>
      <td>0.989205</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>16</td>
      <td>0.020360</td>
      <td>0.042231</td>
      <td>0.986752</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>17</td>
      <td>0.016714</td>
      <td>0.031218</td>
      <td>0.989696</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>18</td>
      <td>0.015005</td>
      <td>0.035987</td>
      <td>0.990186</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>19</td>
      <td>0.013949</td>
      <td>0.033931</td>
      <td>0.989696</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


With a DNN, choosing a large learning rate is unstable. With my 4-layer DNN, setting `lr=1` sometimes gives above 99% but sometimes 50%. Setting a small `lr=0.1` and train for a larger number of epochs gives a more consistent result.
