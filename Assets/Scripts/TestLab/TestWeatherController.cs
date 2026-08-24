using UnityEngine;

public enum TestWeatherType
{
    Dry,
    Rain,
    Snow,
    Fog
}

public class TestWeatherController : MonoBehaviour
{
    [SerializeField] private Transform weatherEffectsRoot;
    [SerializeField] private Light directionalLight;

    public TestWeatherType CurrentWeather { get; private set; } = TestWeatherType.Dry;

    private AutonomousTestVehicleController testVehicle;
    private ParticleSystem rainParticles;
    private ParticleSystem snowParticles;
    private Material rainMaterial;
    private Material snowMaterial;
    private bool originalFog;
    private Color originalFogColor;
    private float originalFogDensity;
    private Color originalAmbientLight;
    private float originalLightIntensity = 1f;

    private void Awake()
    {
        originalFog = RenderSettings.fog;
        originalFogColor = RenderSettings.fogColor;
        originalFogDensity = RenderSettings.fogDensity;
        originalAmbientLight = RenderSettings.ambientLight;
        if (directionalLight != null)
            originalLightIntensity = directionalLight.intensity;

        if (weatherEffectsRoot == null)
        {
            GameObject root = new GameObject("WeatherEffects");
            root.transform.SetParent(transform, false);
            weatherEffectsRoot = root.transform;
        }

        rainParticles = CreateRain();
        snowParticles = CreateSnow();
        SetDry();
    }

    private void Update()
    {
        if (testVehicle == null || weatherEffectsRoot == null)
            return;

        Vector3 vehiclePosition = testVehicle.transform.position;
        weatherEffectsRoot.position = new Vector3(vehiclePosition.x, vehiclePosition.y + 12f, vehiclePosition.z);
    }

    public void SetTestVehicle(AutonomousTestVehicleController vehicle)
    {
        testVehicle = vehicle;
        ApplyVehicleModifiers();
    }

    public void SetDry()
    {
        CurrentWeather = TestWeatherType.Dry;
        StopParticles();
        SetFog(originalFog, originalFogColor, originalFogDensity);
        SetLighting(originalAmbientLight, originalLightIntensity);
        ApplyVehicleModifiers();
    }

    public void SetRain()
    {
        CurrentWeather = TestWeatherType.Rain;
        StopParticles();
        if (rainParticles != null) rainParticles.Play();
        SetFog(true, new Color(0.48f, 0.52f, 0.56f), 0.008f);
        SetLighting(new Color(0.42f, 0.45f, 0.5f), 0.65f);
        ApplyVehicleModifiers();
    }

    public void SetSnow()
    {
        CurrentWeather = TestWeatherType.Snow;
        StopParticles();
        if (snowParticles != null) snowParticles.Play();
        SetFog(true, new Color(0.78f, 0.82f, 0.85f), 0.012f);
        SetLighting(new Color(0.7f, 0.73f, 0.76f), 0.8f);
        ApplyVehicleModifiers();
    }

    public void SetFoggy()
    {
        CurrentWeather = TestWeatherType.Fog;
        StopParticles();
        SetFog(true, new Color(0.65f, 0.68f, 0.7f), 0.035f);
        SetLighting(new Color(0.55f, 0.57f, 0.6f), 0.55f);
        ApplyVehicleModifiers();
    }

    private void ApplyVehicleModifiers()
    {
        if (testVehicle == null)
            return;

        switch (CurrentWeather)
        {
            case TestWeatherType.Rain:
                testVehicle.SetWeatherCruisingFactor(0.90f);
                break;
            case TestWeatherType.Snow:
                testVehicle.SetWeatherCruisingFactor(0.78f);
                break;
            case TestWeatherType.Fog:
                testVehicle.SetWeatherCruisingFactor(0.82f);
                break;
            default:
                testVehicle.SetWeatherCruisingFactor(1.00f);
                break;
        }
    }

    private ParticleSystem CreateRain()
    {
        ParticleSystem particles = CreateParticleObject("RainParticles");
        ParticleSystem.MainModule main = particles.main;
        main.startLifetime = 1.3f;
        main.startSpeed = 18f;
        main.startSize = 0.055f;
        main.startColor = new Color(0.65f, 0.78f, 1f, 0.75f);
        main.gravityModifier = 0.4f;
        main.maxParticles = 3500;

        ParticleSystem.EmissionModule emission = particles.emission;
        emission.rateOverTime = 1400f;
        ConfigureBoxShape(particles, new Vector3(24f, 1f, 24f));

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        renderer.renderMode = ParticleSystemRenderMode.Stretch;
        renderer.lengthScale = 2.5f;
        rainMaterial = CreateParticleMaterial("RoadWeave Rain", new Color(0.65f, 0.78f, 1f, 0.75f));
        renderer.sharedMaterial = rainMaterial;
        particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        return particles;
    }

    private ParticleSystem CreateSnow()
    {
        ParticleSystem particles = CreateParticleObject("SnowParticles");
        ParticleSystem.MainModule main = particles.main;
        main.startLifetime = 7f;
        main.startSpeed = 1.7f;
        main.startSize = new ParticleSystem.MinMaxCurve(0.08f, 0.22f);
        main.startColor = Color.white;
        main.gravityModifier = 0.05f;
        main.maxParticles = 2500;

        ParticleSystem.EmissionModule emission = particles.emission;
        emission.rateOverTime = 500f;
        ConfigureBoxShape(particles, new Vector3(24f, 1f, 24f));

        ParticleSystemRenderer renderer = particles.GetComponent<ParticleSystemRenderer>();
        renderer.renderMode = ParticleSystemRenderMode.Billboard;
        snowMaterial = CreateParticleMaterial("RoadWeave Snow", Color.white);
        renderer.sharedMaterial = snowMaterial;
        particles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        return particles;
    }

    private ParticleSystem CreateParticleObject(string objectName)
    {
        GameObject effect = new GameObject(objectName);
        effect.transform.SetParent(weatherEffectsRoot, false);
        effect.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
        return effect.AddComponent<ParticleSystem>();
    }

    private static void ConfigureBoxShape(ParticleSystem particles, Vector3 size)
    {
        ParticleSystem.ShapeModule shape = particles.shape;
        shape.enabled = true;
        shape.shapeType = ParticleSystemShapeType.Box;
        shape.scale = size;
    }

    private static Material CreateParticleMaterial(string materialName, Color color)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Particles/Unlit");
        if (shader == null) shader = Shader.Find("Particles/Standard Unlit");
        if (shader == null) shader = Shader.Find("Unlit/Color");
        Material material = new Material(shader) { name = materialName, color = color };
        if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
        return material;
    }

    private void StopParticles()
    {
        if (rainParticles != null) rainParticles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
        if (snowParticles != null) snowParticles.Stop(true, ParticleSystemStopBehavior.StopEmittingAndClear);
    }

    private static void SetFog(bool enabled, Color color, float density)
    {
        RenderSettings.fog = enabled;
        RenderSettings.fogMode = FogMode.ExponentialSquared;
        RenderSettings.fogColor = color;
        RenderSettings.fogDensity = density;
    }

    private void SetLighting(Color ambient, float directionalIntensity)
    {
        RenderSettings.ambientLight = ambient;
        if (directionalLight != null)
            directionalLight.intensity = directionalIntensity;
    }

    private void OnDestroy()
    {
        RenderSettings.fog = originalFog;
        RenderSettings.fogColor = originalFogColor;
        RenderSettings.fogDensity = originalFogDensity;
        RenderSettings.ambientLight = originalAmbientLight;
        if (directionalLight != null)
            directionalLight.intensity = originalLightIntensity;
        if (rainMaterial != null) Destroy(rainMaterial);
        if (snowMaterial != null) Destroy(snowMaterial);
    }
}
