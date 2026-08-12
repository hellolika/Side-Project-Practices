CREATE PROCEDURE [dbo].[HRMS_DoClock.2.0.0]
	@memberId INT,
	@date DATE,
	@offSetInMinutes INT,
	@isClockIn BIT,
	@isInCompany BIT,
	@location NVARCHAR(500),
 @clockInRemark NVARCHAR(500),
 @clockOutRemark NVARCHAR(500)
AS
BEGIN
	SET NOCOUNT ON;

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[ReClockRecord] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [Date] = @date AND [IsClockIn] = @isClockIn)
	BEGIN
		SELECT 16 AS ErrorCode;
		RETURN;
	END

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[TakeLeaveRecord] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [IsCancel] = 0 AND [Status] IN(0, 1)
	AND DATEADD(MINUTE, @offSetInMinutes, GETDATE())
		BETWEEN
			-- TAKE LEAVE START FROM MORNING
			CASE WHEN DATEPART(hour, [StartDate]) < 12 
			THEN CAST(CAST([StartDate] AS DATE) AS DATETIME)
			-- TAKE LEAVE START FROM AFTERNOON
			ELSE [StartDate] END
			
			-- TAKE LEAVE END IN THE AFTERNOON
		AND CASE WHEN DATEPART(hour, [EndDate]) > 13
			THEN CONVERT(DATETIME, CONVERT(varchar(10), [EndDate], 120) + ' 23:59:59.000', 120)
			-- TAKE LEAVE END IN THE MORNING 
			ELSE [EndDate] END)
	BEGIN
		SELECT 29 AS ErrorCode;
		RETURN;
	END

	IF(@isInCompany = 1)
	BEGIN
		SET @location = (SELECT [LocationName] FROM [dbo].[LocationDetail] WITH(NOLOCK) 
							WHERE [Id] = (SELECT [WorkLocationId] 
							FROM [dbo].[Member] WITH(NOLOCK) WHERE [Id] = @memberId))
	END

	IF NOT EXISTS (SELECT TOP 1 1 FROM [dbo].[Attendances] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [WorkDate] = @date)
	BEGIN
		DECLARE @clockInTime AS TIME(0) = NULL;
		DECLARE @clockOutTime AS TIME(0) = NULL;
		DECLARE @clockInLocation AS NVARCHAR(500) = NULL;
		DECLARE @clockOutLocation AS NVARCHAR(500) = NULL;
		IF(@isClockIn = 1)
		BEGIN
			SET @clockInTime = GETDATE();
			SET @clockInLocation = @location;
		END
		ELSE
		BEGIN
			SET @clockOutTime = GETDATE();
			SET @clockOutLocation = @location;
		END
		INSERT INTO [dbo].[Attendances] (
				MemberId,
				WorkDate,
				ClockIn,
				ClockOut,
				ClockInLocation,
				ClockOutLocation,
 ClockInRemark,
 ClockOutRemark)
				VALUES(
				@memberId,
				@date,
				@clockInTime,
				@clockOutTime,
				@clockInLocation,
				@clockOutLocation,
 @clockInRemark,
 @clockOutRemark) 
		SELECT 0 AS ErrorCode;
	END
	ELSE IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Attendances] WITH(NOLOCK)
	WHERE [MemberId] = @memberId AND [ClockOut] IS NULL AND [WorkDate] = @date)
	BEGIN
		IF (@isClockIn = 0)
		BEGIN
		--DECLARE @clockin DATETIME 
		--SELECT @clockin = [ClockIn] FROM [dbo].[Attendances] a WITH(NOLOCK) WHERE a.[MemberId] = @memberId
		--	IF (DATEADD(mi,1,@clockin) > GETDATE()) 
		--	BEGIN
		--		SELECT 17;
		--	END
		--	ELSE
			BEGIN
				UPDATE [dbo].[Attendances] 
				SET [ClockOut] = GETDATE() , [ClockOutLocation] = @location, [ClockOutRemark] = @clockOutRemark
				WHERE [MemberId] = @memberId AND [WorkDate] = @date 
				AND [ClockOut] IS NULL
				SELECT 0 AS ErrorCode;
			END
		END
		ELSE
			SELECT 14 AS ErrorCode;
	END	
	ELSE
		SELECT 15 AS ErrorCode;
END
