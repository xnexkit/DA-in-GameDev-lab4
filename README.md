# DA-in-GameDev-lab4
# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #4 выполнил:
- Шмаков Никита Владимирович
- ФО210005
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Ознакомиться с алгоритмом работы "Перцептрона".

## Задание 1
### Реализовать перцептрон, который умеет производить вычисления:

- OR | Дать комментарии о корректности работы

Перцептрон работает при значении эпох равное 4. При меньшем значении периодически возникают ошибки в работе перцептрона.

![изображение](https://user-images.githubusercontent.com/45539357/203804428-e392cad7-5318-4d65-b48f-aaa1d2de7c70.png)

- AND

Перцептрон работает при значении эпох не меньше 5.

![изображение](https://user-images.githubusercontent.com/45539357/203809143-27e2a061-a8c2-47bb-948b-d97bb51692c0.png)

- NAND

Аналогично AND.

![изображение](https://user-images.githubusercontent.com/45539357/203809743-18f15a0b-f9ca-41e3-ba04-48f764a87bb2.png)


- XOR

Не работает совсем при любом значении эпох. `totalError` не превышал 4 и не был ниже 2.

![изображение](https://user-images.githubusercontent.com/45539357/203811825-ed479dd3-99a9-44ac-852b-33f82d94d6e4.png)

## Задание 2
### Построить графики зависимости количества эпох от ошибки обучения. Указать от чего зависит необходимое количество эпох обучения.

- OR

![изображение](https://user-images.githubusercontent.com/45539357/203832445-1001aaa2-a412-4cf0-a0ec-7a68f23307d1.png)

- AND

![изображение](https://user-images.githubusercontent.com/45539357/203833267-852bf35f-bd06-4a28-94f1-470cef13fa6f.png)

- NAND

![изображение](https://user-images.githubusercontent.com/45539357/203833560-c68cd038-ff44-4f30-997c-6b86d7805e81.png)

- XOR

![изображение](https://user-images.githubusercontent.com/45539357/203833710-7b389b47-7478-4b80-9fe3-0d53a95231de.png)


Для уменьшения вероятности появления `totalError` большее нуля необходимо увеличивать количество эпох. При значении эпох равное 8 и более, появление не нулевой ошибки крайне мало. Но для операции `XOR` это правило не работает.

## Задание 3
### Построить визуальную модель работы перцептрона

![Lab4 - SampleScene - Windows, Mac, Linux - Unity 2021 3 9f1_ _DX11_ 2022-11-24 23-28-46](https://user-images.githubusercontent.com/45539357/203848264-c3bdaeba-c1f8-4d09-a2d2-b5b203661447.gif)

Для нижних кубов добавил следующий скипт перцептрона, настроил набор данных для соответсвующих моделей перцептрона (OR, AND):
```cs
using System;
using UnityEngine;
using Random = UnityEngine.Random;

[System.Serializable]
public class TrainingSet
{
	public double[] input;
	public double output;
	}

public class Perceptron : MonoBehaviour
{
	public TrainingSet[] ts;
	public int _epochs = 0;
	double[] weights = { 0, 0 };
	double bias = 0;
	double totalError = 0;

	double DotProductBias(double[] v1, double[] v2)
	{
		if (v1 == null || v2 == null)
			return -1;

		if (v1.Length != v2.Length)
			return -1;

		double d = 0;
		for (int x = 0; x < v1.Length; x++)
		{
			d += v1[x] * v2[x];
		}

		d += bias;

		return d;
	}

	double CalcOutput(int i)
	{
		double dp = DotProductBias(weights, ts[i].input);
		if (dp > 0) return (1);
		return (0);
	}

	void InitialiseWeights()
	{
		for (int i = 0; i < weights.Length; i++)
		{
			weights[i] = Random.Range(-1.0f, 1.0f);
		}
		bias = Random.Range(-1.0f, 1.0f);
	}

	void UpdateWeights(int j)
	{
		double error = ts[j].output - CalcOutput(j);
		totalError += Mathf.Abs((float) error);
		for (int i = 0; i < weights.Length; i++)
		{
			weights[i] = weights[i] + error * ts[j].input[i];
		}
		bias += error;
	}

	double CalcOutput(double i1, double i2)
	{
		double[] inp = new double[] { i1, i2 };
		double dp = DotProductBias(weights, inp);
		if (dp > 0) return (1);
		return (0);
	}

	void Train(int epochs)
	{
		InitialiseWeights();

		for (int e = 0; e < epochs; e++)
		{
			totalError = 0;
			for (int t = 0; t < ts.Length; t++)
			{
				UpdateWeights(t);
				Debug.Log("W1: " + (weights[0]) + " W2: " + (weights[1]) + " B: " + bias);
			}
			Debug.Log("TOTAL ERROR: " + totalError);
		}
	}

	void Start()
	{
		Train(_epochs);
		Debug.Log("Epochs: " + _epochs);
		Debug.Log("Test 0 0: " + CalcOutput(0, 0));
		Debug.Log("Test 0 1: " + CalcOutput(0, 1));
		Debug.Log("Test 1 0: " + CalcOutput(1, 0));
		Debug.Log("Test 1 1: " + CalcOutput(1, 1));
	}

	void Update()
	{
	}

	private void OnCollisionEnter(Collision collision)
	{
		if (collision.gameObject.name == "Plane")
			return;

		var mesh = GetComponent<MeshRenderer>();
		var otherMesh = collision.gameObject.GetComponent<MeshRenderer>();

		var colId = mesh.material.color == Color.black ? 1 : 0;
		var otherColId = otherMesh.material.color == Color.black ? 1 : 0;


		var result = Math.Abs(CalcOutput(colId, otherColId) - 1) < 0.1 ? Color.black : Color.white;
		otherMesh.material.color = result;
		mesh.material.color = result;
	}
}
```

## Выводы

В ходе лабораторной работы я ознакомился с алгоритмом работы перцептрона. Реализовал несколько перцептронов, выполняющих вышеизложенные операции.
Построил графики завимостей количества эпох от ошибки обучения. А также сделал визуальную модель работы перцептрона, демонстрирующую возможное применение перцептрона.

## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**
