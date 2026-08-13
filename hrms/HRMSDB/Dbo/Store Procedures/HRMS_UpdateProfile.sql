CREATE PROCEDURE [dbo].[HRMS_UpdateProfile.1.0.0]
  @memberId INT,
  @teamId INT,
  @workLocationId INT,
  @email NVARCHAR(100),
  @phoneNumber NVARCHAR(50),
  @address NVARCHAR(200),
  @bankAccount NVARCHAR(20),
  @remark NVARCHAR(200),
  @bankName NVARCHAR(200),
  @position NVARCHAR(200)
AS
BEGIN

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [Email] = @email 
	AND [Id] != @memberId AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 26 AS ErrorCode;
		RETURN;
	END

	IF EXISTS (SELECT TOP 1 1 FROM [dbo].[Member] WITH(NOLOCK) WHERE [PhoneNumber] = @phoneNumber
	AND [Id] != @memberId AND ([IsDeleted] = 0 OR [IsDeleted] IS NULL))
	BEGIN
		SELECT 34 AS ErrorCode;
		RETURN;
	END

	IF NOT EXISTS (SELECT TOP 1 1 FROM [dbo].[Team] WITH(NOLOCK) WHERE [TeamId] = @teamId)
	BEGIN
		SELECT 39 AS ErrorCode;
		RETURN;
	END

	UPDATE [dbo].[Member] 
	SET
	TeamId = ISNULL(@teamId, TeamId),
	WorkLocationId = ISNULL(@workLocationId, WorkLocationId),
	Email = ISNULL(@email, Email),
	PhoneNumber = ISNULL(@phoneNumber, PhoneNumber),
	Address = ISNULL(@address, Address),
	BankAccount = ISNULL(@bankAccount, BankAccount),
	Remark = ISNULL(@remark, Remark),
	BankName = ISNULL(@bankName, BankName),
	Position = ISNULL(@position,Position)
	WHERE Id = @memberId

	SELECT 0 as ErrorCode
END